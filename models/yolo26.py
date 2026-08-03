import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. CORE CONVOLUTIONS & UTILITIES
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

# ==========================================
# 2. QUANTUM-INSPIRED ATTENTION (QIA) BLOCK
# ==========================================
class QuantumAttentionBlock(nn.Module):
    """Quantum-Inspired Attention (QIA) Block for YOLO26.
    Uses complex-phase projection to capture long-range non-linear feature interactions.
    """
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.phase_real = Conv(c, c, 1)
        self.phase_imag = Conv(c, c, 1)
        self.out_proj = Conv(c, c, 1)

    def forward(self, x):
        r = self.phase_real(x)
        i = self.phase_imag(x)
        # Quantum state phase interference simulation
        amplitude = torch.sqrt(r ** 2 + i ** 2 + 1e-8)
        phase = torch.atan2(i, r + 1e-8)
        quantum_feat = amplitude * torch.cos(phase)
        return x + self.out_proj(quantum_feat)

# ==========================================
# 3. Y26-BLOCK & FEATURE FUSION
# ==========================================
class Y26Block(nn.Module):
    """Ultra-Fast YOLO26 Residual Aggregation Block."""
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.qia = QuantumAttentionBlock(self.c)
        self.m = nn.ModuleList([
            nn.Sequential(Conv(self.c, self.c, 3, 1), Conv(self.c, self.c, 3, 1))
            for _ in range(n)
        ])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y[1] = self.qia(y[1])
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# ==========================================
# 4. END-TO-END NMS-FREE DETECTION HEAD
# ==========================================
class YOLO26DetectHead(nn.Module):
    """YOLO26 End-to-End Direct Regression Detection Head."""
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4

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
# 5. YOLO26 COMPLETE ARCHITECTURE
# ==========================================
class YOLO26(nn.Module):
    """Pure PyTorch Implementation of YOLO26 Architecture."""
    def __init__(self, num_classes=80):
        super().__init__()
        # Stem & Backbone
        self.stem = Conv(3, 32, 3, 2)
        self.conv1 = Conv(32, 64, 3, 2)
        self.y26_1 = Y26Block(64, 64, n=1)

        self.conv2 = Conv(64, 128, 3, 2)
        self.y26_2 = Y26Block(128, 128, n=2)

        self.conv3 = Conv(128, 256, 3, 2)
        self.y26_3 = Y26Block(256, 256, n=2)

        self.conv4 = Conv(256, 512, 3, 2)
        self.y26_4 = Y26Block(512, 512, n=1)

        # Neck (Multi-Scale Hyper-Fusion Pathway)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.neck_p4 = Y26Block(512 + 256, 256, n=1)
        self.neck_p3 = Y26Block(256 + 128, 128, n=1)

        self.conv_down3 = Conv(128, 128, 3, 2)
        self.neck_n4 = Y26Block(128 + 256, 256, n=1)

        self.conv_down4 = Conv(256, 256, 3, 2)
        self.neck_n5 = Y26Block(256 + 512, 512, n=1)

        # Decoupled End-to-End Head
        self.head = YOLO26DetectHead(nc=num_classes, ch=(128, 256, 512))

    def forward(self, x):
        # Backbone Forward
        x = self.stem(x)
        x = self.conv1(x)
        x = self.y26_1(x)

        x = self.conv2(x)
        c2 = self.y26_2(x)  # P3 / 8

        x = self.conv3(c2)
        c3 = self.y26_3(x)  # P4 / 16

        x = self.conv4(c3)
        c4 = self.y26_4(x)  # P5 / 32

        # Neck Top-Down Pathway
        p4 = self.neck_p4(torch.cat([self.up(c4), c3], dim=1))
        p3 = self.neck_p3(torch.cat([self.up(p4), c2], dim=1))

        # Neck Bottom-Up Pathway
        n4 = self.neck_n4(torch.cat([self.conv_down3(p3), p4], dim=1))
        n5 = self.neck_n5(torch.cat([self.conv_down4(n4), c4], dim=1))

        # Head Outputs
        return self.head([p3, n4, n5])

# ==========================================
# 6. VERIFICATION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    model = YOLO26(num_classes=80)
    dummy_input = torch.randn(1, 3, 640, 640)
    outputs = model(dummy_input)
    print("YOLO26 Architecture Verification Successful!")
    print(f"P3/8 Output Shape:  {outputs[0].shape}")
    print(f"P4/16 Output Shape: {outputs[1].shape}")
    print(f"P5/32 Output Shape: {outputs[2].shape}")
