# =============================
# 1. Импорт библиотек
# =============================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import os

# =============================
# 2. Класс Dataset
# =============================
class GazeDataset(Dataset):
    """
    Пример: Dataset, возвращающий левый/правый глаз, позу головы и метку направления взгляда.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.samples = os.listdir(os.path.join(data_dir, "eyes"))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Пути
        eye_path = os.path.join(self.data_dir, "eyes", self.samples[idx])
        headpose_path = os.path.join(self.data_dir, "headpose", self.samples[idx].replace(".jpg", ".npy"))
        gaze_path = os.path.join(self.data_dir, "gaze", self.samples[idx].replace(".jpg", ".npy"))

        # Загрузка данных
        eye_img = cv2.imread(eye_path)
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
        if self.transform:
            eye_img = self.transform(eye_img)

        headpose = np.load(headpose_path)  # (yaw, pitch, roll)
        gaze = np.load(gaze_path)          # (theta, phi)

        return eye_img, torch.tensor(headpose, dtype=torch.float32), torch.tensor(gaze, dtype=torch.float32)


# =============================
# 3. Архитектура модели
# =============================
class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()
        # CNN backbone для глаз
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(512, 128)

        # Полносвязная часть для позы головы
        self.head_fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Объединение глаз + головы
        self.fc_combined = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # θ и φ
        )

    def forward(self, eye_img, head_pose):
        eye_feat = self.cnn(eye_img)
        head_feat = self.head_fc(head_pose)
        combined = torch.cat([eye_feat, head_feat], dim=1)
        gaze = self.fc_combined(combined)
        return gaze


# =============================
# 4. Угловая функция потерь
# =============================
def angular_loss(pred, target):
    """
    Угол между предсказанным и реальным направлением взгляда в радианах.
    """
    pred_norm = nn.functional.normalize(pred, dim=1)
    target_norm = nn.functional.normalize(target, dim=1)
    dot_product = (pred_norm * target_norm).sum(dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    angle = torch.acos(dot_product)
    return angle.mean()


# =============================
# 5. Обучение
# =============================
def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for eye, head, gaze in dataloader:
        eye, head, gaze = eye.to(device), head.to(device), gaze.to(device)
        optimizer.zero_grad()
        pred = model(eye, head)
        loss = angular_loss(pred, gaze)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# =============================
# 6. Основная часть
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = GazeDataset("dataset_path", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = GazeNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    loss = train_model(model, dataloader, optimizer, device)
    print(f"Epoch {epoch+1} | Angular loss: {loss:.4f}")

torch.save(model.state_dict(), "gaze_estimator.pth")
print("✅ Model saved successfully!")
