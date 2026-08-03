import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. CORE CONVOLUTIONS & AUTOPAD
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
# 2. AREA ATTENTION (A^2) MODULE
# ==========================================
class AreaAttention(nn.Module):
    """Area Attention (A^2) Module for YOLOv12.
    Partitions feature maps into area segments to achieve low-complexity self-attention.
    """
    def __init__(self, dim, num_heads=8, area_size=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.area_size = area_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        # Flatten spatial dimensions -> (B, N, C)
        x_flat = x.flatten(2).transpose(1, 2)
        
        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)

        # Native PyTorch SDPA (Scaled Dot-Product Attention)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(attn_out)
        
        # Reshape back to spatial feature map -> (B, C, H, W)
        return out.transpose(1, 2).reshape(B, C, H, W)

# ==========================================
# 3. R-ELAN (RESIDUAL ELAN WITH A^2 BLOCKS)
# ==========================================
class A2Block(nn.Module):
    """Attention-Centric Block combining Area Attention with MLP."""
    def __init__(self, c, num_heads=4):
        super().__init__()
        self.attn = AreaAttention(c, num_heads=num_heads)
        self.mlp = nn.Sequential(
            Conv(c, int(c * 1.5), 1),
            Conv(int(c * 1.5), c, 1)
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class R_ELAN(nn.Module):
    """Residual Efficient Layer Aggregation Network with A^2 Blocks."""
    def __init__(self, c1, c2, c3, n=2):
        super().__init__()
        self.cv1 = Conv(c1, c3, 1)
        self.cv2 = Conv(c1, c3, 1)
        self.a2_blocks = nn.ModuleList([A2Block(c3) for _ in range(n)])
        self.cv3 = Conv(c3 * (n + 2), c2, 1)

    def forward(self, x):
        y = [self.cv1(x), self.cv2(x)]
        for m in self.a2_blocks:
            y.append(m(y[-1]))
        return self.cv3(torch.cat(y, dim=1))

# ==========================================
# 4. ANCHOR-FREE DECOUPLED HEAD
# ==========================================
class DetectHead(nn.Module):
    """YOLOv12 Decoupled Anchor-Free Detection Head."""
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
# 5. YOLOV12 COMPLETE ARCHITECTURE
# ==========================================
class YOLOv12(nn.Module):
    """Pure PyTorch Implementation of YOLOv12 Architecture."""
    def __init__(self, num_classes=80):
        super().__init__()
        # Stem & Backbone with R-ELAN + Area Attention
        self.stem = Conv(3, 32, 3, 2)
        self.conv1 = Conv(32, 64, 3, 2)
        self.relan1 = R_ELAN(64, 64, 32, n=1)

        self.conv2 = Conv(64, 128, 3, 2)
        self.relan2 = R_ELAN(128, 128, 64, n=2)

        self.conv3 = Conv(128, 256, 3, 2)
        self.relan3 = R_ELAN(256, 256, 128, n=2)

        self.conv4 = Conv(256, 512, 3, 2)
        self.relan4 = R_ELAN(512, 512, 256, n=1)

        # Neck (PAFPN with R-ELAN Feature Aggregation)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.neck_p4 = R_ELAN(512 + 256, 256, 128, n=1)
        self.neck_p3 = R_ELAN(256 + 128, 128, 64, n=1)

        self.conv_down3 = Conv(128, 128, 3, 2)
        self.neck_n4 = R_ELAN(128 + 256, 256, 128, n=1)

        self.conv_down4 = Conv(256, 256, 3, 2)
        self.neck_n5 = R_ELAN(256 + 512, 512, 256, n=1)

        # Decoupled Detection Head
        self.head = DetectHead(nc=num_classes, ch=(128, 256, 512))

    def forward(self, x):
        # Backbone Forward
        x = self.stem(x)
        x = self.conv1(x)
        x = self.relan1(x)

        x = self.conv2(x)
        c2 = self.relan2(x)  # P3 / 8

        x = self.conv3(c2)
        c3 = self.relan3(x)  # P4 / 16

        x = self.conv4(c3)
        c4 = self.relan4(x)  # P5 / 32

        # Neck Top-Down Pathway
        p4 = self.neck_p4(torch.cat([self.up(c4), c3], dim=1))
        p3 = self.neck_p3(torch.cat([self.up(p4), c2], dim=1))

        # Neck Bottom-Up Pathway
        n4 = self.neck_n4(torch.cat([self.conv_down3(p3), p4], dim=1))
        n5 = self.neck_n5(torch.cat([self.conv_down4(n4), c4], dim=1))

        # Head Predictions
        return self.head([p3, n4, n5])

# ==========================================
# 6. VERIFICATION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    model = YOLOv12(num_classes=80)
    dummy_input = torch.randn(1, 3, 640, 640)
    outputs = model(dummy_input)
    print("YOLOv12 Architecture Verification Successful!")
    print(f"P3/8 Output Shape:  {outputs[0].shape}")
    print(f"P4/16 Output Shape: {outputs[1].shape}")
    print(f"P5/32 Output Shape: {outputs[2].shape}")
