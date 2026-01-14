#!/usr/bin/env python3
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    path: str
    label: int


def list_videos(root: str) -> Tuple[List[Sample], List[str]]:
    root_p = Path(root)
    class_names = sorted([p.name for p in root_p.iterdir() if p.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    samples: List[Sample] = []
    for c in class_names:
        for vid in (root_p / c).rglob("*.mp4"):
            samples.append(Sample(str(vid), class_to_idx[c]))

    if len(samples) == 0:
        raise RuntimeError(f"No mp4 files found under {root}")

    return samples, class_names


def _read_frame_at(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def load_clip_opencv(video_path: str, num_frames: int, size: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length <= 0:
        cap.release()
        raise RuntimeError(f"Could not read frame count: {video_path}")

    # Uniform indices
    if length >= num_frames:
        idxs = np.linspace(0, length - 1, num_frames).astype(int)
    else:
        # Pad by repeating last
        idxs = np.arange(length).tolist() + [length - 1] * (num_frames - length)
        idxs = np.array(idxs, dtype=int)

    frames = []
    for i in idxs:
        fr = _read_frame_at(cap, int(i))
        if fr is None:
            fr = frames[-1] if len(frames) > 0 else np.zeros((size, size, 3), dtype=np.uint8)
        fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_LINEAR)
        frames.append(fr)

    cap.release()
    clip = np.stack(frames, axis=0)  # [T, H, W, C]
    return clip


class VideoFolderDataset(Dataset):
    def __init__(self, split_root: str, num_frames: int, input_size: int):
        self.samples, self.class_names = list_videos(split_root)
        self.num_frames = num_frames
        self.input_size = input_size

        # VideoMAE default normalization is like ImageNet
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        clip = load_clip_opencv(s.path, self.num_frames, self.input_size)  # [T,H,W,C], uint8
        clip = clip.astype(np.float32) / 255.0
        clip = (clip - self.mean) / self.std

        # to torch: [T,C,H,W]
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2).contiguous()
        label = torch.tensor(s.label, dtype=torch.long)
        return clip, label
