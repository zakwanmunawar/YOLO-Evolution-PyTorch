import torch
import torch.nn as nn

# ==========================================
# 1. CORE BUILDING BLOCKS (Conv & Autopad)
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

# ==========================================
# 2. E-ELAN & TRANSITION BLOCKS
# ==========================================
class E_ELAN(nn.Module):
    """Extended Efficient Layer Aggregation Network (E-ELAN)."""
    def __init__(self, c1, c2, c3, n=4):
        super().__init__()
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c1, c3, 1, 1)
        self.cv3 = nn.ModuleList([Conv(c3, c3, 3, 1) for _ in range(n)])
        self.cv4 = Conv(c3 * (n + 2), c2, 1, 1)

    def forward(self, x):
        y = [self.cv1(x), self.cv2(x)]
        for m in self.cv3:
            y.append(m(y[-1]))
        return self.cv4(torch.cat(y, 1))

class MP(nn.Module):
    """Max Pooling Block."""
    def __init__(self, k=2):
        super().__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)

class Transition_Block(nn.Module):
    """Combined MaxPool + Conv downsampling block."""
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = Conv(c1, c2 // 2, 1, 1)
        self.cv2 = Conv(c2 // 2, c2 // 2, 3, 2)
        self.cv3 = Conv(c1, c2 // 2, 1, 1)
        self.mp = MP(k=2)

    def forward(self, x):
        path1 = self.cv2(self.cv1(x))
        path2 = self.cv3(self.mp(x))
        return torch.cat([path1, path2], 1)

# ==========================================
# 3. YOLOV7 COMPLETE ARCHITECTURE
# ==========================================
class YOLOv7(nn.Module):
    """Pure PyTorch Implementation of YOLOv7."""
    def __init__(self, num_classes=80):
        super().__init__()
        self.nc = num_classes
        self.no = 3 * (5 + num_classes)  # 3 anchors per grid cell

        # Stem Layer
        self.stem1 = Conv(3, 32, 3, 1)
        self.stem2 = Conv(32, 64, 3, 2)
        self.stem3 = Conv(64, 64, 3, 1)

        # Backbone (Transition + E-ELAN Stages)
        self.down1 = Transition_Block(64, 128)
        self.elan1 = E_ELAN(128, 256, 64)

        self.down2 = Transition_Block(256, 256)
        self.elan2 = E_ELAN(256, 512, 128)

        self.down3 = Transition_Block(512, 512)
        self.elan3 = E_ELAN(512, 1024, 256)

        self.down4 = Transition_Block(1024, 1024)
        self.elan4 = E_ELAN(1024, 1024, 256)

        # Neck (PAFPN with E-ELAN Blocks)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce_p5 = Conv(1024, 256, 1, 1)
        self.reduce_p4 = Conv(1024, 256, 1, 1)

        self.neck_elan1 = E_ELAN(512, 256, 128)
        self.reduce_p4_feat = Conv(256, 128, 1, 1)
        self.reduce_c3 = Conv(512, 128, 1, 1)

        self.neck_elan2 = E_ELAN(256, 128, 64)

        # Multi-scale Detection Heads
        self.head_p3 = nn.Conv2d(128, self.no, 1)
        self.head_p4 = nn.Conv2d(256, self.no, 1)
        self.head_p5 = nn.Conv2d(256, self.no, 1)

    def forward(self, x):
        # Backbone Forward
        x = self.stem1(x)
        x = self.stem2(x)
        x = self.stem3(x)

        x = self.down1(x)
        c2 = self.elan1(x)

        x = self.down2(c2)
        c3 = self.elan2(x)

        x = self.down3(c3)
        c4 = self.elan3(x)

        x = self.down4(c4)
        c5 = self.elan4(x)

        # Feature Pyramid Fusion (Neck)
        p5_up = self.up(self.reduce_p5(c5))
        p4_cat = torch.cat([p5_up, self.reduce_p4(c4)], dim=1)
        p4_feat = self.neck_elan1(p4_cat)

        p4_up = self.up(self.reduce_p4_feat(p4_feat))
        p3_cat = torch.cat([p4_up, self.reduce_c3(c3)], dim=1)
        p3_feat = self.neck_elan2(p3_cat)

        # Head Predictions
        out_p3 = self.head_p3(p3_feat)
        out_p4 = self.head_p4(p4_feat)
        out_p5 = self.head_p5(self.reduce_p5(c5))

        return [out_p3, out_p4, out_p5]

# ==========================================
# 4. VERIFICATION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    model = YOLOv7(num_classes=80)
    dummy_input = torch.randn(1, 3, 640, 640)
    outputs = model(dummy_input)
    print("YOLOv7 Architecture Verification Successful!")
    print(f"P3 Output Shape: {outputs[0].shape}")
    print(f"P4 Output Shape: {outputs[1].shape}")
    print(f"P5 Output Shape: {outputs[2].shape}")
