import torch
import torch.nn as nn

# ==========================================
# 1. CORE CONVOLUTIONS & DEPTHWISE SEPARABLE
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

class DSConv(nn.Module):
    """Depthwise Separable Convolution Module (Lightweight Feature Extractor)."""
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.dw = Conv(c1, c1, k=k, s=s, g=c1)  # Depthwise conv
        self.pw = Conv(c1, c2, k=1, s=1)        # Pointwise conv

    def forward(self, x):
        return self.pw(self.dw(x))

# ==========================================
# 2. HYPERACE (HYPERGRAPH ADAPTIVE CORRELATION)
# ==========================================
class HyperACE(nn.Module):
    """Hypergraph-based Adaptive Correlation Enhancement (HyperACE) Module.
    Captures multi-to-multi high-order spatial and semantic correlations across features.
    """
    def __init__(self, dim, num_edges=8):
        super().__init__()
        self.dim = dim
        self.num_edges = num_edges
        
        # Adaptive Incidence Matrix Generation
        self.incidence_proj = nn.Conv2d(dim, num_edges, kernel_size=1)
        self.node_proj = Conv(dim, dim, 1)
        self.hyperedge_proj = Conv(dim, dim, 1)
        self.out_proj = Conv(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        # Generate soft incidence matrix H_mat -> (B, num_edges, N)
        H_mat = torch.softmax(self.incidence_proj(x).view(B, self.num_edges, N), dim=-1)
        
        # Node features -> (B, N, C)
        X_nodes = self.node_proj(x).view(B, C, N).permute(0, 2, 1)
        
        # Node to Hyperedge Aggregation: E = H_mat @ X_nodes -> (B, num_edges, C)
        E_edges = torch.bmm(H_mat, X_nodes)
        
        # Hyperedge Feature Transformation
        E_edges = self.hyperedge_proj(E_edges.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
        
        # Hyperedge to Node Distribution: X_out = H_mat^T @ E -> (B, N, C)
        X_enhanced = torch.bmm(H_mat.permute(0, 2, 1), E_edges)
        
        X_out = X_enhanced.permute(0, 2, 1).view(B, C, H, W)
        return x + self.out_proj(X_out)

# ==========================================
# 3. DS-C3K2 & FULLPAD FEATURE BLOCKS
# ==========================================
class DS_Bottleneck(nn.Module):
    """Depthwise-Separable Bottleneck."""
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = DSConv(c1, c2, k=3, s=1)
        self.cv2 = DSConv(c2, c2, k=3, s=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class DS_C3k2(nn.Module):
    """C3k2 with Depthwise Separable Convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList([DS_Bottleneck(self.c, self.c, shortcut) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# ==========================================
# 4. ANCHOR-FREE DECOUPLED HEAD
# ==========================================
class DetectHead(nn.Module):
    """YOLOv13 Anchor-Free Decoupled Detection Head."""
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
# 5. YOLOV13 COMPLETE ARCHITECTURE
# ==========================================
class YOLOv13(nn.Module):
    """Pure PyTorch Implementation of YOLOv13 Architecture."""
    def __init__(self, num_classes=80):
        super().__init__()
        # Stem & Backbone with DS-C3k2 and HyperACE High-Order Correlation
        self.stem = Conv(3, 32, 3, 2)
        self.conv1 = Conv(32, 64, 3, 2)
        self.dsc3k2_1 = DS_C3k2(64, 64, n=1)

        self.conv2 = Conv(64, 128, 3, 2)
        self.dsc3k2_2 = DS_C3k2(128, 128, n=2)

        self.conv3 = Conv(128, 256, 3, 2)
        self.dsc3k2_3 = DS_C3k2(256, 256, n=2)

        self.conv4 = Conv(256, 512, 3, 2)
        self.dsc3k2_4 = DS_C3k2(512, 512, n=1)
        self.hyperace = HyperACE(512, num_edges=8)  # Global Hypergraph Correlation

        # Neck (FullPAD Paradigm with HyperACE Fusion)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.neck_p4 = DS_C3k2(512 + 256, 256, n=1)
        self.neck_p3 = DS_C3k2(256 + 128, 128, n=1)

        self.conv_down3 = Conv(128, 128, 3, 2)
        self.neck_n4 = DS_C3k2(128 + 256, 256, n=1)

        self.conv_down4 = Conv(256, 256, 3, 2)
        self.neck_n5 = DS_C3k2(256 + 512, 512, n=1)

        # Decoupled Detection Head
        self.head = DetectHead(nc=num_classes, ch=(128, 256, 512))

    def forward(self, x):
        # Backbone Forward
        x = self.stem(x)
        x = self.conv1(x)
        x = self.dsc3k2_1(x)

        x = self.conv2(x)
        c2 = self.dsc3k2_2(x)  # P3 / 8

        x = self.conv3(c2)
        c3 = self.dsc3k2_3(x)  # P4 / 16

        x = self.conv4(c3)
        c4 = self.dsc3k2_4(x)
        c5 = self.hyperace(c4) # P5 / 32 with HyperACE

        # Neck Top-Down Pathway
        p4 = self.neck_p4(torch.cat([self.up(c5), c3], dim=1))
        p3 = self.neck_p3(torch.cat([self.up(p4), c2], dim=1))

        # Neck Bottom-Up Pathway
        n4 = self.neck_n4(torch.cat([self.conv_down3(p3), p4], dim=1))
        n5 = self.neck_n5(torch.cat([self.conv_down4(n4), c5], dim=1))

        # Head Predictions
        return self.head([p3, n4, n5])

# ==========================================
# 6. VERIFICATION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    model = YOLOv13(num_classes=80)
    dummy_input = torch.randn(1, 3, 640, 640)
    outputs = model(dummy_input)
    print("YOLOv13 Architecture Verification Successful!")
    print(f"P3/8 Output Shape:  {outputs[0].shape}")
    print(f"P4/16 Output Shape: {outputs[1].shape}")
    print(f"P5/32 Output Shape: {outputs[2].shape}")
