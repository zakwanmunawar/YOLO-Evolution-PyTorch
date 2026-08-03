import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 0. DOWNLOAD SAMPLE IMAGE IF MISSING
# ==========================================
if not os.path.exists("real_cat.jpg"):
    os.system('curl -L -o real_cat.jpg "https://picsum.photos/id/219/416/416"')

# ==========================================
# 1. MODEL ARCHITECTURE (YOLOv2 + Reorg)
# ==========================================
class ReorgLayer(nn.Module):
    """Passthrough layer to stack fine-grained feature maps."""
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        _s = self.stride
        out_height = height // _s
        out_width = width // _s

        x = x.view(batch_size, channels, out_height, _s, out_width, _s)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(batch_size, channels * _s * _s, out_height, out_width)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class YOLOv2TinyTiny(nn.Module):
    def __init__(self, num_classes=2, num_anchors=5):
        super(YOLOv2TinyTiny, self).__init__()
        self.num_outputs = num_anchors * (num_classes + 5)

        self.stage1 = nn.Sequential(
            ConvBlock(3, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, 3, padding=1)
        )

        self.route_stage = nn.Sequential(
            ConvBlock(128, 256, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512, 3, padding=1)
        )

        self.reorg = ReorgLayer(stride=2)
        self.final_stage = ConvBlock(512 + 512, 1024, 3, padding=1)
        self.output_conv = nn.Conv2d(1024, self.num_outputs, 1, 1, 0)

    def forward(self, x):
        feat_early = self.stage1(x)
        feat_deep = self.route_stage(feat_early)
        passthrough = self.reorg(feat_early)

        x = torch.cat((passthrough, feat_deep), dim=1)
        x = self.final_stage(x)
        output = self.output_conv(x)
        return output

# ==========================================
# 2. DATASET PREPARATION
# ==========================================
class YOLOv2SingleRealImageDataset(Dataset):
    def __init__(self, image_path="real_cat.jpg", grid_size=52, num_anchors=5, num_classes=2):
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.image_path = image_path

    def __len__(self):
        return 10  # 10 iterations for quick execution check

    def __getitem__(self, idx):
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        num_outputs_per_anchor = self.num_classes + 5
        target_tensor = torch.zeros((self.num_anchors * num_outputs_per_anchor, self.grid_size, self.grid_size))

        cx, cy, w, h = 0.5, 0.5, 0.65, 0.75
        grid_x, grid_y = int(cx * self.grid_size), int(cy * self.grid_size)
        class_id = 0  # Class 0 = Cat

        for anchor_idx in range(self.num_anchors):
            start_channel = anchor_idx * num_outputs_per_anchor
            target_tensor[start_channel, grid_y, grid_x] = cx * self.grid_size - grid_x
            target_tensor[start_channel + 1, grid_y, grid_x] = cy * self.grid_size - grid_y
            target_tensor[start_channel + 2, grid_y, grid_x] = w
            target_tensor[start_channel + 3, grid_y, grid_x] = h
            target_tensor[start_channel + 4, grid_y, grid_x] = 1.0
            target_tensor[start_channel + 5 + class_id, grid_y, grid_x] = 1.0

        return image_tensor, target_tensor

# ==========================================
# 3. TRAINING PIPELINE
# ==========================================
def run_yolov2_real_test(dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing pipeline on device: {device}\n")

    NUM_CLASSES = 2
    NUM_ANCHORS = 5
    BATCH_SIZE = 2
    EPOCHS = 1

    model = YOLOv2TinyTiny(num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS).to(device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    print(f"Starting actual image training run loop over {len(dataset)} items...")

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for batch_idx, (images, targets) in enumerate(dataloader):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"Batch [{batch_idx+1}/{len(dataloader)}] Complete -> Loss: {loss.item():.4f}")

        print(f"\nEpoch [{epoch+1}/{EPOCHS}] Run Complete. Mean loss value tracked: {running_loss/len(dataloader):.4f}")

# ==========================================
# 4. VISUALIZATION
# ==========================================
def plot_real_cat_sample(dataset):
    image_tensor, _ = dataset[0]
    img_numpy = image_tensor.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img_numpy)

    w_box, h_box = 0.65 * 416, 0.75 * 416
    xmin, ymin = (0.5 * 416) - (w_box / 2), (0.5 * 416) - (h_box / 2)

    rect = patches.Rectangle((xmin, ymin), w_box, h_box, linewidth=3, edgecolor="lime", facecolor="none")
    ax.add_patch(rect)

    plt.text(xmin, ymin - 10, "Cat (Class 0)", color="white", weight="bold", bbox=dict(facecolor="black", alpha=0.5, pad=2))
    plt.title("YOLOv2 Pipeline Real Photo Target Check")
    plt.axis("off")
    plt.show()

# ==========================================
# 5. EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    dataset = YOLOv2SingleRealImageDataset()
    run_yolov2_real_test(dataset)
    plot_real_cat_sample(dataset)
