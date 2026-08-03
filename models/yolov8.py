import torch
import torch.nn as nn

# ==========================================
# 1. CORE BUILDING BLOCKS (Conv & Bottleneck)
# ==========================================
def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    """Standard Convolution + BatchNorm + SiLU activation."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """Standard Bottleneck with optional residual shortcut."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# ==========================================
# 2. C2F & SPPF MODULES
# ==========================================
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList([Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# ==========================================
# 3. ANCHOR-FREE DECOUPLED HEAD
# ==========================================
class DetectHead(nn.Module):
    """YOLOv8 Anchor-Free Decoupled Detection Head."""
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers (3 scales)
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # outputs per anchor

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max((ch[0], min((self.nc, 100))))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )

    def forward(self, x):
        for i in range(self.nl):
            bbox = self.cv2[i](x[i])
            cls = self.cv3[i](x[i])
            x[i] = torch.cat((bbox, cls), 1)
        return x

# ==========================================
# 4. YOLOV8 COMPLETE ARCHITECTURE
# ==========================================
class YOLOv8(nn.Module):
    """Pure PyTorch Implementation of YOLOv8 Architecture."""
    def __init__(self, num_classes=80):
        super().__init__()
        # Backbone (Stem + Downsampling + C2f + SPPF)
        self.stem = Conv(3, 32, 3, 2)
        self.conv1 = Conv(32, 64, 3, 2)
        self.c2f1 = C2f(64, 64, n=1, shortcut=True)

        self.conv2 = Conv(64, 128, 3, 2)
        self.c2f2 = C2f(128, 128, n=2, shortcut=True)

        self.conv3 = Conv(128, 256, 3, 2)
        self.c2f3 = C2f(256, 256, n=2, shortcut=True)

        self.conv4 = Conv(256, 512, 3, 2)
        self.c2f4 = C2f(512, 512, n=1, shortcut=True)
        self.sppf = SPPF(512, 512, k=5)

        # Neck (PANet with C2f blocks)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f_p4 = C2f(512 + 256, 256, n=1, shortcut=False)
        self.c2f_p3 = C2f(256 + 128, 128, n=1, shortcut=False)

        self.conv_down3 = Conv(128, 128, 3, 2)
        self.c2f_n4 = C2f(128 + 256, 256, n=1, shortcut=False)

        self.conv_down4 = Conv(256, 256, 3, 2)
        self.c2f_n5 = C2f(256 + 512, 512, n=1, shortcut=False)

        # Anchor-Free Decoupled Detection Head
        self.head = DetectHead(nc=num_classes, ch=(128, 256, 512))

    def forward(self, x):
        # Backbone
        x = self.stem(x)
        x = self.conv1(x)
        x = self.c2f1(x)

        x = self.conv2(x)
        c2 = self.c2f2(x)  # P3 / 8

        x = self.conv3(c2)
        c3 = self.c2f3(x)  # P4 / 16

        x = self.conv4(c3)
        c4 = self.c2f4(x)
        c5 = self.sppf(c4) # P5 / 32

        # Neck Top-Down Pathway
        p4 = self.c2f_p4(torch.cat([self.up(c5), c3], dim=1))
        p3 = self.c2f_p3(torch.cat([self.up(p4), c2], dim=1))

        # Neck Bottom-Up Pathway
        n4 = self.c2f_n4(torch.cat([self.conv_down3(p3), p4], dim=1))
        n5 = self.c2f_n5(torch.cat([self.conv_down4(n4), c5], dim=1))

        # Decoupled Head Outputs across 3 Scales
        return self.head([p3, n4, n5])

# ==========================================
# 5. VERIFICATION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    model = YOLOv8(num_classes=80)
    dummy_input = torch.randn(1, 3, 640, 640)
    outputs = model(dummy_input)
    print("YOLOv8 Architecture Verification Successful!")
    print(f"P3/8 Output Shape:  {outputs[0].shape}")
    print(f"P4/16 Output Shape: {outputs[1].shape}")
    print(f"P5/32 Output Shape: {outputs[2].shape}")
