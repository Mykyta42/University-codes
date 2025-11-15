"""
MPIIGaze gaze direction training script with tqdm progress bars.
Predicts 3D gaze vector from left/right eye crops + head pose.
"""

import os
import cv2
import math
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.models as models
import torch.optim as optim

# ----------------------------
# 1) Dataset
# ----------------------------
class MPIIGazeDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform_eye=None,
        crop_scale=0.18,
        max_subjects=None,     # limit participants number
        max_days_per_subject=None,  # limit days number per participant
        max_samples=None       # limit samples number
    ):
        """
            root_dir: path to MPIIGaze root
            transform_eye: eye torchvision transforms
            crop_scale: eye crop scale
            max_subjects: imit participants number
            max_days_per_subject: limit days number per participant
            max_samples: limit samples number
        """
        self.root_dir = root_dir
        self.transform_eye = transform_eye
        self.crop_scale = crop_scale
        self.samples = []

        data_orig = os.path.join(root_dir, "Data", "Original")
        if not os.path.isdir(data_orig):
            raise RuntimeError(f"Directory not found: {data_orig}")

        subjects = sorted([s for s in os.listdir(data_orig) if os.path.isdir(os.path.join(data_orig, s))])
        if max_subjects is not None:
            subjects = subjects[:max_subjects]

        total_count = 0
        for subj_idx, subj in enumerate(subjects):
            subj_path = os.path.join(data_orig, subj)
            day_dirs = sorted([d for d in os.listdir(subj_path) if os.path.isdir(os.path.join(subj_path, d)) and d[:3] == "day"])
            if max_days_per_subject is not None:
                day_dirs = day_dirs[:max_days_per_subject]

            for day in day_dirs:
                day_path = os.path.join(subj_path, day)
                ann_path = os.path.join(day_path, "annotation.txt")
                if not os.path.isfile(ann_path):
                    continue

                img_files = sorted([f for f in os.listdir(day_path) if f.lower().endswith(('.jpg', '.png'))])
                with open(ann_path, "r") as fh:
                    lines = [l.strip() for l in fh.readlines() if l.strip()]

                n = min(len(img_files), len(lines))
                for i in range(n):
                    parts = lines[i].split()
                    try:
                        arr = [float(x) for x in parts]
                    except ValueError:
                        continue
                    if len(arr) < 41:
                        continue

                    lm2d = np.array(arr[0:24], dtype=np.float32).reshape(-1, 2)
                    gaze3d = np.array(arr[26:29], dtype=np.float32)
                    head6 = np.array(arr[29:35], dtype=np.float32)
                    img_path = os.path.join(day_path, img_files[i])

                    if os.path.isfile(img_path):
                        self.samples.append({
                            "img": img_path,
                            "lm2d": lm2d,
                            "gaze3d": gaze3d,
                            "head6": head6
                        })
                        total_count += 1

                    if max_samples is not None and total_count >= max_samples:
                        break
                if max_samples is not None and total_count >= max_samples:
                    break
            if max_samples is not None and total_count >= max_samples:
                break

        print(f"[INFO] MPIIGaze: loaded {len(self.samples)} samples "
              f"(subjects={len(subjects)}, limit={max_samples})")

    def __len__(self):
        return len(self.samples)

    def crop_eye_by_landmarks(self, img, lm2d, which='left'):
        h, w = img.shape[:2]
        cx = w / 2.0
        left_pts = lm2d[lm2d[:,0] < cx]
        right_pts = lm2d[lm2d[:,0] >= cx]

        if which == 'left':
            pts = left_pts if len(left_pts) > 0 else right_pts
        else:
            pts = right_pts if len(right_pts) > 0 else left_pts

        if len(pts) == 0:
            cx_px, cy_px = w/2, h/2
        else:
            cx_px = np.mean(pts[:,0])
            cy_px = np.mean(pts[:,1])

        side = int(max(40, min(w,h) * self.crop_scale))
        half = side // 2
        x1 = int(max(0, cx_px - half))
        y1 = int(max(0, cy_px - half))
        x2 = int(min(w, cx_px + half))
        y2 = int(min(h, cy_px + half))

        if (x2 - x1) < side:
            if x1 == 0: x2 = min(w, x1 + side)
            else: x1 = max(0, x2 - side)
        if (y2 - y1) < side:
            if y1 == 0: y2 = min(h, y1 + side)
            else: y1 = max(0, y2 - side)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return pil

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s["img"])
        if img is None:
            raise RuntimeError(f"Can't read image: {s['img']}")
        lm2d = s["lm2d"]

        eye_left = self.crop_eye_by_landmarks(img, lm2d, which='left')
        eye_right = self.crop_eye_by_landmarks(img, lm2d, which='right')

        if self.transform_eye:
            eye_left_t = self.transform_eye(eye_left)
            eye_right_t = self.transform_eye(eye_right)
        else:
            transform_default = T.Compose([T.Resize((224,224)), T.ToTensor()])
            eye_left_t = transform_default(eye_left)
            eye_right_t = transform_default(eye_right)

        head6 = torch.tensor(s["head6"], dtype=torch.float32)
        gaze3d = torch.tensor(s["gaze3d"], dtype=torch.float32)

        return eye_left_t, eye_right_t, head6, gaze3d


# ----------------------------
# 2) Model
# ----------------------------
class DualEyeGazeNet(nn.Module):
    def __init__(self, pretrained=True, head_feat_dim=64):
        super().__init__()
        base = models.resnet18(pretrained=pretrained)
        modules = list(base.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        feat_dim = 512

        self.eye_fc = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )

        self.head_fc = nn.Sequential(
            nn.Linear(6, head_feat_dim), nn.ReLU(),
            nn.Linear(head_feat_dim, head_feat_dim), nn.ReLU()
        )

        self.comb_fc = nn.Sequential(
            nn.Linear(128*2 + head_feat_dim, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, eye_left, eye_right, head6):
        fl = self.feature_extractor(eye_left).view(eye_left.size(0), -1)
        fr = self.feature_extractor(eye_right).view(eye_right.size(0), -1)
        fl = self.eye_fc(fl)
        fr = self.eye_fc(fr)
        h = self.head_fc(head6)
        combined = torch.cat([fl, fr, h], dim=1)
        out = self.comb_fc(combined)
        return out


# ----------------------------
# 3) Loss
# ----------------------------
def angular_loss_3d(pred_vec, target_vec, eps=1e-7):
    pred_n = F.normalize(pred_vec, dim=1, eps=eps)
    tgt_n = F.normalize(target_vec, dim=1, eps=eps)
    dot = torch.clamp(torch.sum(pred_n * tgt_n, dim=1), -1.0, 1.0)
    ang = torch.acos(dot)
    return ang.mean(), ang


# ----------------------------
# 4) Train / Eval with tqdm progress bars
# ----------------------------
def train_epoch(model, loader, optimizer, device, epoch_idx, total_epochs):
    model.train()
    total_loss = 0.0
    total_ang = 0.0
    n = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch_idx}/{total_epochs} [Train]", leave=False)
    for left, right, head6, gaze3d in pbar:
        left = left.to(device); right = right.to(device)
        head6 = head6.to(device); gaze3d = gaze3d.to(device)
        optimizer.zero_grad()
        pred = model(left, right, head6)
        loss_mean, ang = angular_loss_3d(pred, gaze3d)
        loss_mean.backward()
        optimizer.step()

        bs = left.size(0)
        total_loss += loss_mean.item() * bs
        total_ang += ang.sum().item()
        n += bs
        avg_ang_deg = (total_ang / n) * 180 / math.pi
        pbar.set_postfix({"avg_deg": f"{avg_ang_deg:.2f}"})
    return total_loss / n, (total_ang / n)


@torch.no_grad()
def eval_epoch(model, loader, device, epoch_idx, total_epochs):
    model.eval()
    total_loss = 0.0
    total_ang = 0.0
    n = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch_idx}/{total_epochs} [Eval]", leave=False)
    for left, right, head6, gaze3d in pbar:
        left = left.to(device); right = right.to(device)
        head6 = head6.to(device); gaze3d = gaze3d.to(device)
        pred = model(left, right, head6)
        loss_mean, ang = angular_loss_3d(pred, gaze3d)
        bs = left.size(0)
        total_loss += loss_mean.item() * bs
        total_ang += ang.sum().item()
        n += bs
        avg_ang_deg = (total_ang / n) * 180 / math.pi
        pbar.set_postfix({"avg_deg": f"{avg_ang_deg:.2f}"})
    return total_loss / n, (total_ang / n)


# ----------------------------
# 5) Plot helper
# ----------------------------
def plot_errors(train_deg, val_deg, out_path=None):
    epochs = range(1, len(train_deg)+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_deg, marker='o', label='Train angular error (deg)')
    plt.plot(epochs, val_deg, marker='s', label='Val angular error (deg)')
    plt.xlabel('Epoch')
    plt.ylabel('Angular error (degrees)')
    plt.title('Angular error per epoch')
    plt.grid(True)
    plt.legend()
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    plt.show()


# ----------------------------
# 6) Main
# ----------------------------
if __name__ == "__main__":
    ROOT = "MPIIGaze"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_eye = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    dataset = MPIIGazeDataset(ROOT, transform_eye=transform_eye, crop_scale=0.18, max_days_per_subject=1)
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = DualEyeGazeNet(pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    EPOCHS = 10
    train_deg_history = []
    val_deg_history = []

    for epoch in range(1, EPOCHS+1):
        train_loss_rad, train_ang_rad = train_epoch(model, train_loader, optimizer, device, epoch, EPOCHS)
        val_loss_rad, val_ang_rad = eval_epoch(model, val_loader, device, epoch, EPOCHS)

        train_deg = (train_ang_rad * 180.0 / math.pi)
        val_deg = (val_ang_rad * 180.0 / math.pi)
        train_deg_history.append(train_deg)
        val_deg_history.append(val_deg)

        print(f"Epoch {epoch}/{EPOCHS} | Train {train_deg:.3f}° | Val {val_deg:.3f}°")


    torch.save(model.state_dict(), "gazenet.pth")
    plot_errors(train_deg_history, val_deg_history, out_path="angular_error.png")
    print("Training finished.")
