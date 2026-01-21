#!/usr/bin/env python3
# train_mobilenet_video.py
#
# MobileNetV3-Large backbone + temporal conv head for binary video classification
# Logs JSONL compatible with your visualize_training.py script:
#   outputs/<exp>/logs/train.log            ({"step","train_loss",...})
#   outputs/<exp>/results/val_metrics.jsonl ({"epoch","loss","acc","f1","confusion_matrix", "tp","tn","fp","fn",...})

import os
import json
import time
import random
from dataclasses import dataclass
from pathlib import Path
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
    H: int = 192
    W: int = 192

    # Frame sampling inside a clip
    stride: int = 1

    # Training
    batch_size: int = 8
    # On Jetson, PyAV can segfault in multi-worker dataloaders; default to 0.
    num_workers: int = 0
    epochs: int = 20
    lr_backbone: float = 1e-4
    lr_head: float = 5e-4
    weight_decay: float = 1e-4

    # Class imbalance (set to approx #neg/#pos in TRAIN split)
    pos_weight: float = 2.5

    # Mixed precision (only active on CUDA)
    amp: bool = True

    # Prediction threshold for metrics
    threshold: float = 0.5
    debug_checks: bool = False

    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
    video_extensions: Tuple[str, ...] = (".mp4", ".mov", ".avi")


# -------------------------
# JSONL logging helpers
# -------------------------
def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def binary_confusion_counts(logits: torch.Tensor, y: torch.Tensor, threshold: float = 0.5):
    """
    logits: [B,1] raw
    y:      [B,1] float {0,1}
    returns tp, tn, fp, fn (python ints)
    """
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).to(torch.int64)
    yt = (y >= 0.5).to(torch.int64)

    tp = ((pred == 1) & (yt == 1)).sum().item()
    tn = ((pred == 0) & (yt == 0)).sum().item()
    fp = ((pred == 1) & (yt == 0)).sum().item()
    fn = ((pred == 0) & (yt == 1)).sum().item()
    return tp, tn, fp, fn


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

    Returns:
      x: [T, 3, H, W]
      y: [1] float (0.0 or 1.0)
    """
    def __init__(
        self,
        split_dir: str,
        T: int,
        H: int,
        W: int,
        stride: int,
        exts: Tuple[str, ...],
        video_exts: Tuple[str, ...],
        train: bool,
    ):
        self.split_dir = split_dir
        self.T = T
        self.H = H
        self.W = W
        self.stride = stride
        self.exts = exts
        self.video_exts = video_exts
        self.train = train

        # samples: (payload, label, kind)
        #   kind="video": payload=str path
        #   kind="frames": payload=List[str] frame paths
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
            transforms.Normalize(mean=mean, std=std),
        ])

        # Mild augmentations (train only)
        self.aug_tf = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        ])

    def _preprocess_frame(self, img: torch.Tensor) -> torch.Tensor:
        img = img.to(torch.float32)
        if img.numel() > 0:
            finite_max = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).max()
            if finite_max > 1.5:
                img = img / 255.0
        img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        img = self.base_tf(img)
        img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        return img

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
            video, _, _ = read_video(sample, pts_unit="sec")  # [Tv, H, W, C]
            if video.numel() == 0:
                raise RuntimeError(f"Empty video: {sample}")
            inds = self._sample_indices(video.shape[0])
            for i in inds:
                img = video[i].permute(2, 0, 1)  # [C, H, W]
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                elif img.shape[0] > 3:
                    img = img[:3]

                img = self._preprocess_frame(img)
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

                img = self._preprocess_frame(img)
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
def evaluate(model, loader, device, threshold: float = 0.5):
    model.eval()
    crit = nn.BCEWithLogitsLoss()

    total = 0
    loss_sum = 0.0
    tp = tn = fp = fn = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if hasattr(model, "cfg") and getattr(model.cfg, "debug_checks", False):
            if not torch.isfinite(x).all():
                raise ValueError("Non-finite values in eval inputs.")
        logits = model(x)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        if hasattr(model, "cfg") and getattr(model.cfg, "debug_checks", False):
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite values in eval logits.")
        loss = crit(logits, y)

        loss_sum += loss.item() * x.size(0)
        total += x.size(0)

        b_tp, b_tn, b_fp, b_fn = binary_confusion_counts(logits, y, threshold=threshold)
        tp += b_tp
        tn += b_tn
        fp += b_fp
        fn += b_fn

    acc = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    # confusion matrix format expected by your visualizer:
    # [[TN, FP],
    #  [FN, TP]]
    cm = [[int(tn), int(fp)],
          [int(fn), int(tp)]]

    return {
        "loss": safe_div(loss_sum, max(total, 1)),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": cm,
        "n": int(total),
        "threshold": float(threshold),
    }


# -------------------------
# Training
# -------------------------
def train():
    cfg = CFG()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Experiment directory compatible with visualize_training.py defaults
    exp_dir = Path(f"outputs/exp_mobilenet_T{cfg.T}_{cfg.H}")
    logs_path = exp_dir / "logs" / "train.log"
    val_path = exp_dir / "results" / "val_metrics.jsonl"
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2))

    # Clear old logs if they exist
    for p in [logs_path, val_path]:
        if p.exists():
            p.unlink()

    train_dir = os.path.join(cfg.root, "train")
    val_dir = os.path.join(cfg.root, "val")

    train_ds = ClipFolderDataset(
        train_dir, cfg.T, cfg.H, cfg.W, cfg.stride,
        cfg.extensions, cfg.video_extensions, train=True
    )
    val_ds = ClipFolderDataset(
        val_dir, cfg.T, cfg.H, cfg.W, cfg.stride,
        cfg.extensions, cfg.video_extensions, train=False
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
    model.cfg = cfg

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

    # AMP setup (works across torch versions)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device.type, enabled=(cfg.amp and device.type == "cuda"))

        def autocast_ctx():
            return torch.amp.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda"))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

        def autocast_ctx():
            return torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda"))

    best_val_f1 = -1.0
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if cfg.debug_checks and not torch.isfinite(x).all():
                raise ValueError("Non-finite values in training inputs.")

            opt.zero_grad(set_to_none=True)

            with autocast_ctx():
                logits = model(x)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                if cfg.debug_checks and not torch.isfinite(logits).all():
                    raise ValueError("Non-finite values in training logits.")
                loss = crit(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            running += loss.item() * bs
            total += bs

            global_step += 1
            append_jsonl(logs_path, {
                "time": time.time(),
                "step": int(global_step),
                "epoch": int(epoch),
                "train_loss": float(loss.item()),
                "batch_size": int(bs),
            })

        train_loss = running / max(total, 1)

        val_metrics = evaluate(model, val_loader, device, threshold=cfg.threshold)

        append_jsonl(val_path, {
            "time": time.time(),
            "epoch": int(epoch),
            "loss": float(val_metrics["loss"]),
            "acc": float(val_metrics["acc"]),
            "f1": float(val_metrics["f1"]),
            "precision": float(val_metrics["precision"]),
            "recall": float(val_metrics["recall"]),
            "tp": int(val_metrics["tp"]),
            "tn": int(val_metrics["tn"]),
            "fp": int(val_metrics["fp"]),
            "fn": int(val_metrics["fn"]),
            "confusion_matrix": val_metrics["confusion_matrix"],
            "n": int(val_metrics["n"]),
            "threshold": float(val_metrics["threshold"]),
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.3f} | "
            f"val_f1={val_metrics['f1']:.3f} | "
            f"TP={val_metrics['tp']} TN={val_metrics['tn']} FP={val_metrics['fp']} FN={val_metrics['fn']}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {"model": model.state_dict(), "cfg": cfg.__dict__, "best_val_f1": best_val_f1},
                ckpt_dir / "best.pt"
            )

    print("Best val_f1:", best_val_f1)
    print("Logs written to:", logs_path)
    print("Val metrics written to:", val_path)
    print("Checkpoint written to:", ckpt_dir / "best.pt")


if __name__ == "__main__":
    train()
