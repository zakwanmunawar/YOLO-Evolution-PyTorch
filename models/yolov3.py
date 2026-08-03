import torch
import torch.nn as nn

# ==========================================
# 1. CORE BUILDING BLOCKS (Darknet-53)
# ==========================================

class ConvBlock(nn.Module):
    """Standard Convolution + BatchNorm + LeakyReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """Standard Darknet Residual Block."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels // 2, kernel_size=1),
            ConvBlock(channels // 2, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

# ==========================================
# 2. COMPLETE YOLOV3 MODEL ASSEMBLY
# ==========================================

class YOLOv3(nn.Module):
    """Pure PyTorch Implementation of YOLOv3 (Darknet-53 Backbone + FPN Head)."""
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.out_channels = 3 * (5 + num_classes)

        # Darknet-53 Backbone
        self.p1 = ConvBlock(3, 32, 3, stride=1, padding=1)
        self.p2 = nn.Sequential(
            ConvBlock(32, 64, 3, stride=2, padding=1),
            ResidualBlock(64)
        )
        self.p3 = nn.Sequential(
            ConvBlock(64, 128, 3, stride=2, padding=1),
            *[ResidualBlock(128) for _ in range(2)]
        )
        self.p4 = nn.Sequential(
            ConvBlock(128, 256, 3, stride=2, padding=1),
            *[ResidualBlock(256) for _ in range(8)]
        )
        self.p5 = nn.Sequential(
            ConvBlock(256, 512, 3, stride=2, padding=1),
            *[ResidualBlock(512) for _ in range(8)]
        )
        self.p6 = nn.Sequential(
            ConvBlock(512, 1024, 3, stride=2, padding=1),
            *[ResidualBlock(1024) for _ in range(4)]
        )

        # Deep Scale Head
        self.head_p5 = nn.Sequential(
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3, padding=1),
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3, padding=1),
            ConvBlock(1024, 512, 1)
        )
        self.out_p5 = nn.Sequential(
            ConvBlock(512, 1024, 3, padding=1),
            nn.Conv2d(1024, self.out_channels, 1)
        )

        # Medium Scale Head
        self.up_p5_to_p4 = ConvBlock(512, 256, 1)
        self.head_p4 = nn.Sequential(
            ConvBlock(512 + 256, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3, padding=1),
            ConvBlock(512, 256, 1)
        )
        self.out_p4 = nn.Sequential(
            ConvBlock(256, 512, 3, padding=1),
            nn.Conv2d(512, self.out_channels, 1)
        )

        # Small Scale Head
        self.up_p4_to_p3 = ConvBlock(256, 128, 1)
        self.head_p3 = nn.Sequential(
            ConvBlock(256 + 128, 128, 1),
            ConvBlock(128, 256, 3, padding=1),
            ConvBlock(256, 128, 1),
            ConvBlock(128, 256, 3, padding=1),
            ConvBlock(256, 128, 1)
        )
        self.out_p3 = nn.Sequential(
            ConvBlock(128, 256, 3, padding=1),
            nn.Conv2d(256, self.out_channels, 1)
        )

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        c3 = self.p3(x)
        c4 = self.p4(c3)
        c5 = self.p5(c4)
        c6 = self.p6(c5)

        p5_feat = self.head_p5(c6)
        out_p5 = self.out_p5(p5_feat)

        p5_up = self.up(self.up_p5_to_p4(p5_feat))
        p4_feat = self.head_p4(torch.cat([p5_up, c5], dim=1))
        out_p4 = self.out_p4(p4_feat)

        p4_up = self.up(self.up_p4_to_p3(p4_feat))
        p3_feat = self.head_p3(torch.cat([p4_up, c4], dim=1))
        out_p3 = self.out_p3(p3_feat)

        return [out_p3, out_p4, out_p5]

if __name__ == "__main__":
    model = YOLOv3(num_classes=80)
    dummy_input = torch.randn(1, 3, 416, 416)
    outputs = model(dummy_input)
    print("YOLOv3 Verification Success!")
    print(f"P3 Output Scale: {outputs[0].shape}")
    print(f"P4 Output Scale: {outputs[1].shape}")
    print(f"P5 Output Scale: {outputs[2].shape}")
