
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image, read_video
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


# -------------------------
# Config
# -------------------------
@dataclass
class CFG:
    # Expect this folder structure (mp4):
    # root/
    #   train/
    #     launch/*.mp4
    #     no_launch/*.mp4
    #   val/
    #     launch/*.mp4
    #     no_launch/*.mp4
    #
    # Legacy frame-folder structure is still supported:
    # root/
    #   train/
    #     launch/<clip_id>/*.jpg
    #     no_launch/<clip_id>/*.jpg
    root: str = "./data/dataset"

    T: int = 8
    H: int = 160
    W: int = 160

    # Frame sampling inside a clip
    stride: int = 1

    # Training
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 20
    lr_backbone: float = 1e-4
    lr_head: float = 5e-4
    weight_decay: float = 1e-4

    # Class imbalance (set to approx #neg/#pos in TRAIN split)
    pos_weight: float = 2.5

    # Mixed precision (only active on CUDA)
    amp: bool = True

    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
    video_extensions: Tuple[str, ...] = (".mp4", ".mov", ".avi")


# -------------------------
# Helpers
# -------------------------
def list_frames(folder: str, exts: Tuple[str, ...]) -> List[str]:
    files = [
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f.lower())[1] in exts
    ]
    files.sort()
    return files


def list_videos(folder: str, exts: Tuple[str, ...]) -> List[str]:
    files = [
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f.lower())[1] in exts
    ]
    files.sort()
    return files


# -------------------------
# Dataset
# -------------------------
class ClipFolderDataset(Dataset):
    """
    Reads clips from:
      split_dir/launch/*.mp4
      split_dir/no_launch/*.mp4

    (Legacy) or frame folders:
      split_dir/launch/<clip_id>/*
      split_dir/no_launch/<clip_id>/*

    Each clip is a folder containing ordered frame images.
    Returns:
      x: [T, 3, H, W]
      y: [1] float (0.0 or 1.0)
    """
    def __init__(self, split_dir: str, T: int, H: int, W: int, stride: int,
                 exts: Tuple[str, ...], video_exts: Tuple[str, ...], train: bool):
        self.split_dir = split_dir
        self.T = T
        self.H = H
        self.W = W
        self.stride = stride
        self.exts = exts
        self.video_exts = video_exts
        self.train = train

        self.samples: List[Tuple[object, int, str]] = []

        for label_name, y in [("launch", 1), ("no_launch", 0)]:
            class_dir = os.path.join(split_dir, label_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(
                    f"Expected folder not found: {class_dir}\n"
                    f"Your folder structure should be:\n"
                    f"{split_dir}/launch/*.mp4 and {split_dir}/no_launch/*.mp4"
                )

            video_files = list_videos(class_dir, self.video_exts)
            if video_files:
                for video_path in video_files:
                    self.samples.append((video_path, y, "video"))
            else:
                for clip_id in os.listdir(class_dir):
                    clip_dir = os.path.join(class_dir, clip_id)
                    if not os.path.isdir(clip_dir):
                        continue
                    frames = list_frames(clip_dir, exts)
                    if len(frames) >= self._min_frames_needed():
                        self.samples.append((frames, y, "frames"))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid clips found under: {split_dir}")

        # ImageNet normalization for pretrained backbone (fallback for older torchvision)
        mean = getattr(MobileNet_V3_Large_Weights.DEFAULT, "meta", {}).get(
            "mean", (0.485, 0.456, 0.406)
        )
        std = getattr(MobileNet_V3_Large_Weights.DEFAULT, "meta", {}).get(
            "std", (0.229, 0.224, 0.225)
        )

        self.base_tf = transforms.Compose([
            transforms.Resize((H, W), antialias=True),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=mean, std=std),
        ])

        # Mild augmentations (train only)
        self.aug_tf = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        ])

    def _min_frames_needed(self) -> int:
        # Need enough frames to take T with stride
        # indices: start + i*stride for i in [0..T-1]
        return 1 + (self.T - 1) * self.stride

    def __len__(self):
        return len(self.samples)

    def _sample_indices(self, n_frames: int) -> List[int]:
        need = self._min_frames_needed()
        max_start = n_frames - need
        if max_start < 0:
            if n_frames <= 0:
                return [0] * self.T
            if n_frames == 1:
                return [0] * self.T
            if self.T == 1:
                return [0]
            # Evenly sample with replacement when video is short
            return [
                int(round(i * (n_frames - 1) / (self.T - 1)))
                for i in range(self.T)
            ]

        if self.train:
            start = random.randint(0, max_start)
        else:
            start = max_start // 2  # deterministic "middle" window

        return [start + i * self.stride for i in range(self.T)]

    def __getitem__(self, idx: int):
        sample, y, kind = self.samples[idx]
        clip = []

        if kind == "video":
            video, _, _ = read_video(sample)  # [T, H, W, C]
            if video.numel() == 0:
                raise RuntimeError(f"Empty video: {sample}")
            inds = self._sample_indices(video.shape[0])
            for i in inds:
                img = video[i].permute(2, 0, 1)  # [C, H, W]
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                elif img.shape[0] > 3:
                    img = img[:3]

                img = self.base_tf(img)
                if self.train:
                    img = self.aug_tf(img)
                clip.append(img)
        else:
            frames = sample
            inds = self._sample_indices(len(frames))
            for i in inds:
                img = read_image(frames[i])  # uint8, [C,H,W]
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                elif img.shape[0] > 3:
                    img = img[:3]

                img = self.base_tf(img)
                if self.train:
                    img = self.aug_tf(img)
                clip.append(img)

        x = torch.stack(clip, dim=0)  # [T, 3, H, W]
        y = torch.tensor([float(y)], dtype=torch.float32)  # [1]
        return x, y


# -------------------------
# Model
# -------------------------
class TemporalConvHead(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.dw = nn.Conv1d(C, C, kernel_size=3, padding=1, groups=C, bias=False)
        self.pw = nn.Conv1d(C, C, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(C)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, Fbtc: torch.Tensor) -> torch.Tensor:
        # Fbtc: [B, T, C]
        x = Fbtc.transpose(1, 2)  # [B, C, T]
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = F.silu(x)
        x = self.drop(x)
        x = x.mean(dim=-1)        # temporal average -> [B, C]
        return x


class EdgeVideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT
        backbone = mobilenet_v3_large(weights=weights)

        self.backbone_features = backbone.features
        self.backbone_pool = nn.AdaptiveAvgPool2d(1)

        # MobileNetV3-Large final feature channels
        self.C = 960

        self.temporal = TemporalConvHead(self.C)
        self.classifier = nn.Linear(self.C, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 3, H, W]
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        f = self.backbone_features(x)          # [B*T, 960, h, w]
        f = self.backbone_pool(f).flatten(1)   # [B*T, 960]
        f = f.reshape(B, T, self.C)            # [B, T, 960]

        v = self.temporal(f)                   # [B, 960]
        logit = self.classifier(v)             # [B, 1]
        return logit


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    crit = nn.BCEWithLogitsLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logit = model(x)
        loss = crit(logit, y)

        # accumulate per-sample loss
        loss_sum += loss.item() * x.size(0)

        # predictions
        pred = (torch.sigmoid(logit) >= 0.5).float()
        correct += (pred == y).sum().item()
        total += x.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


# -------------------------
# Training
# -------------------------
def train():
    cfg = CFG()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dir = os.path.join(cfg.root, "train")
    val_dir = os.path.join(cfg.root, "val")

    train_ds = ClipFolderDataset(
        train_dir, cfg.T, cfg.H, cfg.W, cfg.stride, cfg.extensions, cfg.video_extensions, train=True
    )
    val_ds = ClipFolderDataset(
        val_dir, cfg.T, cfg.H, cfg.W, cfg.stride, cfg.extensions, cfg.video_extensions, train=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda")
    )

    model = EdgeVideoModel().to(device)

    # Parameter groups: backbone vs head
    backbone_params = list(model.backbone_features.parameters())
    head_params = list(model.temporal.parameters()) + list(model.classifier.parameters())

    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.lr_backbone},
        {"params": head_params, "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)

    # Imbalance handling (TRAIN only)
    pos_weight = torch.tensor([cfg.pos_weight], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                logit = model(x)
                loss = crit(logit, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item() * x.size(0)
            total += x.size(0)

        train_loss = running / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, "checkpoints/best.pt")

    print("Best val_acc:", best_val_acc)


if __name__ == "__main__":
    train()
