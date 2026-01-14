#!/usr/bin/env python3
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import compute_classification_metrics, metrics_to_dict


@torch.no_grad()
def run_eval(model, loader: DataLoader, device: str) -> Dict:
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0.0
    n = 0

    loss_fn = torch.nn.CrossEntropyLoss()

    for clips, labels in tqdm(loader, desc="eval", leave=False):
        clips = clips.to(device, non_blocking=True)   # [B,T,C,H,W]
        labels = labels.to(device, non_blocking=True)

        out = model(clips)
        logits = out.logits if hasattr(out, "logits") else out

        loss = loss_fn(logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        n += labels.size(0)

        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    m = compute_classification_metrics(y_true, y_pred)
    return {
        "loss": total_loss / max(1, n),
        **metrics_to_dict(m),
    }
