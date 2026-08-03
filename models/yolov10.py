import torch
import torch.nn as nn

# ==========================================
# 1. CORE CONVOLUTION & DOWNSAMPLING BLOCKS
# ==========================================
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    """Standard Conv + BatchNorm + SiLU."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SCDown(nn.Module):
    """Spatial-Channel Decoupled Downsampling Block."""
    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2)  # Depthwise conv for spatial reduction

    def forward(self, x):
        return self.cv2(self.cv1(x))

# ==========================================
# 2. CIB (Compact Inverted Bottleneck) & PSA
# ==========================================
class CIB(nn.Module):
    """Compact Inverted Bottleneck with depthwise convolutions."""
    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        super().__init__()
        c_ = int(c2 * e)
        k_size = 7 if lk else 3  # Large-kernel dwconv optional
        self.cv1 = Conv(c1, c1, 3, 1, g=c1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, k_size, 1, g=c_)
        self.cv4 = Conv(c_, c2, 1, 1)
        self.cv5 = Conv(c2, c2, 3, 1, g=c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        res = x
        x = self.cv5(self.cv4(self.cv3(self.cv2(self.cv1(x)))))
        return res + x if self.add else x

class PSA(nn.Module):
    """Partial Self-Attention Module."""
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1, 1)
        self.attn = nn.Sequential(
            Conv(self.c, self.c, 1, 1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x1, x2 = self.cv1(x).chunk(2, 1)
        # Apply partial self-attention to x2
        b, c, h, w = x2.shape
        attn_map = self.attn(x2)
        x2 = x2 * attn_map
        return self.cv2(torch.cat([x1, x2], dim=1))

# ==========================================
# 3. YOLOV10 NMS-FREE DUAL DETECTION HEAD
# ==========================================
class YOLOv10v10Detect(nn.Module):
    """Dual Assignment Head (One-to-Many + One-to-One for NMS-Free Training)."""
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4

        # One-to-Many Head (Used during training)
        self.cv2_many = nn.ModuleList(
            nn.Sequential(Conv(x, x, 3, g=x), Conv(x, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3_many = nn.ModuleList(
            nn.Sequential(Conv(x, x, 3, g=x), Conv(x, self.nc, 1)) for x in ch
        )

        # One-to-One Head (Used for NMS-free End-to-End Inference)
        self.cv2_one = nn.ModuleList(
            nn.Sequential(Conv(x, x, 3, g=x), Conv(x, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3_one = nn.ModuleList(
            nn.Sequential(Conv(x, x, 3, g=x), Conv(x, self.nc, 1)) for x in ch
        )

    def forward(self, x):
        one2one_outs = []
        for i in range(self.nl):
            bbox_one = self.cv2_one[i](x[i])
            cls_one = self.cv3_one[i](x[i])
            one2one_outs.append(torch.cat((bbox_one, cls_one), 1))

        if self.training:
            one2many_outs = []
            for i in range(self.nl):
                bbox_many = self.cv2_many[i](x[i])
                cls_many = self.cv3_many[i](x[i])
                one2many_outs.append(torch.cat((bbox_many, cls_many), 1))
            return {"one2one": one2one_outs, "one2many": one2many_outs}
        
        return one2one_outs

# ==========================================
# 4. YOLOV10 COMPLETE ARCHITECTURE
# ==========================================
class YOLOv10(nn.Module):
    """Pure PyTorch Implementation of YOLOv10 Architecture."""
    def __init__(self, num_classes=80):
        super().__init__()
        # Stem & Backbone
        self.stem = Conv(3, 32, 3, 2)
        self.down1 = Conv(32, 64, 3, 2)
        self.cib1 = CIB(64, 64, shortcut=True)

        self.down2 = SCDown(64, 128)
        self.cib2 = CIB(128, 128, shortcut=True)

        self.down3 = SCDown(128, 256)
        self.cib3 = CIB(256, 256, shortcut=True)

        self.down4 = SCDown(256, 512)
        self.cib4 = CIB(512, 512, shortcut=True, lk=True)
        self.psa = PSA(512, 512)

        # Neck (PANet with SCDown & CIB)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.neck_cib_p4 = CIB(512 + 256, 256, shortcut=False)
        self.neck_cib_p3 = CIB(256 + 128, 128, shortcut=False)

        self.scdown_n4 = SCDown(128, 128)
        self.neck_cib_n4 = CIB(128 + 256, 256, shortcut=False)

        self.scdown_n5 = SCDown(256, 256)
        self.neck_cib_n5 = CIB(256 + 512, 512, shortcut=False)

        # Dual Assignment Detection Head
        self.head = YOLOv10v10Detect(nc=num_classes, ch=(128, 256, 512))

    def forward(self, x):
        # Backbone Forward
        x = self.stem(x)
        x = self.down1(x)
        x = self.cib1(x)

        x = self.down2(x)
        c2 = self.cib2(x)  # P3 / 8

        x = self.down3(c2)
        c3 = self.cib3(x)  # P4 / 16

        x = self.down4(c3)
        c4 = self.cib4(x)
        c5 = self.psa(c4)  # P5 / 32

        # Neck Top-Down Pathway
        p4 = self.neck_cib_p4(torch.cat([self.up(c5), c3], dim=1))
        p3 = self.neck_cib_p3(torch.cat([self.up(p4), c2], dim=1))

        # Neck Bottom-Up Pathway
        n4 = self.neck_cib_n4(torch.cat([self.scdown_n4(p3), p4], dim=1))
        n5 = self.neck_cib_n5(torch.cat([self.scdown_n5(n4), c5], dim=1))

        # Head Outputs
        return self.head([p3, n4, n5])

# ==========================================
# 5. VERIFICATION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    model = YOLOv10(num_classes=80)
    model.eval()  # Set to eval mode for end-to-end inference
    dummy_input = torch.randn(1, 3, 640, 640)
    outputs = model(dummy_input)
    print("YOLOv10 Architecture Verification Successful!")
    print(f"P3/8 Output Shape:  {outputs[0].shape}")
    print(f"P4/16 Output Shape: {outputs[1].shape}")
    print(f"P5/32 Output Shape: {outputs[2].shape}")
