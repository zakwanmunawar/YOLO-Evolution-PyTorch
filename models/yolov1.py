import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 1. SETTINGS & HYPERPARAMETERS
# ==========================================
S, B, C = 7, 2, 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LEARNING_RATE = 2e-5

# ==========================================
# 2. VOC DATASET TARGET ENCODER
# ==========================================
def voc_transform(img, anno):
    target = torch.zeros((S, S, C + B * 5))
    w_img, h_img = img.size
    classes = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

    annotations = anno['annotation']['object']
    if not isinstance(annotations, list): 
        annotations = [annotations]

    for obj in annotations:
        cls_id = classes.index(obj['name'])
        box = obj['bndbox']
        x1, y1, x2, y2 = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])
        x_c, y_c = (x1 + x2) / (2 * w_img), (y1 + y2) / (2 * h_img)
        w, h = (x2 - x1) / w_img, (y2 - y1) / h_img

        i, j = int(S * y_c), int(S * x_c)
        i, j = min(i, S - 1), min(j, S - 1)

        if target[i, j, 20] == 0:
            target[i, j, 20] = 1 # Object Confidence
            target[i, j, 21:25] = torch.tensor([x_c, y_c, w, h])
            target[i, j, cls_id] = 1 # One-hot Class Encoding

    img_tensor = transforms.Compose([
        transforms.Resize((448, 448)), 
        transforms.ToTensor()
    ])(img)
    return img_tensor, target.view(-1)

# ==========================================
# 3. YOLOV1 NEURAL NETWORK ARCHITECTURE
# ==========================================
class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, 3, padding=1), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 128, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)), 
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 4096), nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5))
        )

    def forward(self, x): 
        return self.fc(self.conv(x))

# ==========================================
# 4. CUSTOM YOLOV1 MULTI-PART LOSS
# ==========================================
class SimpleYOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(SimpleYOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5.0

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        target = target.reshape(-1, self.S, self.S, self.C + self.B * 5)

        exists_box = target[..., 20].unsqueeze(3)

        # Coordinate Bounding Box Loss
        p_box = predictions[..., 21:25]
        t_box = target[..., 21:25]
        p_box[..., 2:4] = torch.sign(p_box[..., 2:4]) * torch.sqrt(torch.abs(p_box[..., 2:4] + 1e-6))
        t_box[..., 2:4] = torch.sqrt(t_box[..., 2:4])
        box_loss = self.mse(exists_box * p_box, exists_box * t_box)

        # Object Confidence Loss
        object_loss = self.mse(exists_box * predictions[..., 20:21], exists_box * target[..., 20:21])

        # No-Object Confidence Loss
        no_object_loss = self.mse((1 - exists_box) * predictions[..., 20:21], (1 - exists_box) * target[..., 20:21])

        # Classification Loss
        class_loss = self.mse(exists_box * predictions[..., :20], exists_box * target[..., :20])

        total_loss = (self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_object_loss + class_loss)
        return total_loss

# ==========================================
# 5. VISUALIZATION / INFERENCE FUNCTION
# ==========================================
def predict_and_show(model, dataset, index=15):
    model.eval()
    img_tensor, _ = dataset[index]

    with torch.no_grad():
        prediction = model(img_tensor.unsqueeze(0).to(DEVICE))
        prediction = prediction.view(7, 7, 30).cpu()

    confidences = prediction[:, :, 20]
    best_cell = torch.argmax(confidences)
    row, col = best_cell // 7, best_cell % 7

    box = prediction[row, col, 21:25]
    x_c, y_c, w, h = box[0].item(), box[1].item(), box[2].item(), box[3].item()

    x1 = (x_c - w / 2) * 448
    y1 = (y_c - h / 2) * 448
    width, height = w * 448, h * 448

    img_np = img_tensor.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(img_np)
    rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.title(f"Confidence: {confidences[row, col]:.2f}")
    plt.axis('off')
    plt.show()

# ==========================================
# 6. TRAINING SCRIPT ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    print("Loading VOC Dataset...")
    train_data = datasets.VOCDetection(
        root='./data', year='2012', image_set='train', download=True, transforms=voc_transform
    )
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    model = YOLOv1().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = SimpleYOLOLoss()

    print("\nStarting Training with Custom YOLO Loss...")
    for epoch in range(5):
        for i, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            preds = model(imgs)
            loss = loss_fn(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

    print("\nRunning Inference Sample...")
    predict_and_show(model, train_data, index=15)
