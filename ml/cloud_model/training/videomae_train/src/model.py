#!/usr/bin/env python3
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class R3D18Fallback(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        from torchvision.models.video import r3d_18
        self.backbone = r3d_18(weights=None)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C,H,W] -> r3d expects [B,C,T,H,W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.backbone(x)


def build_model(cfg: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    model_name = cfg.get("model_name", "videomae")
    num_classes = int(cfg["num_classes"])

    preprocess_info = {
        "expects": "B,T,C,H,W",
    }

    if model_name == "videomae":
        try:
            from transformers import VideoMAEForVideoClassification
        except Exception as e:
            print(f"[WARN] transformers VideoMAE not available, falling back to r3d_18. Reason: {e}")
            model = R3D18Fallback(num_classes=num_classes)
            return model, preprocess_info

        hf_model_id = cfg.get("hf_model_id", "MCG-NJU/videomae-base")
        config = None
        input_size = cfg.get("input_size", None)
        num_frames = cfg.get("num_frames", None)
        if input_size is not None or num_frames is not None:
            from transformers import VideoMAEConfig
            config = VideoMAEConfig.from_pretrained(hf_model_id)
            if input_size is not None:
                config.image_size = int(input_size)
            if num_frames is not None:
                config.num_frames = int(num_frames)

        if config is not None:
            config.num_labels = num_classes
            model = VideoMAEForVideoClassification.from_pretrained(
                hf_model_id,
                config=config,
                ignore_mismatched_sizes=True,
            )
        else:
            model = VideoMAEForVideoClassification.from_pretrained(
                hf_model_id,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )

        # Optional dropout override
        dropout = cfg.get("dropout", None)
        if dropout is not None:
            if hasattr(model.config, "hidden_dropout_prob"):
                model.config.hidden_dropout_prob = float(dropout)
            if hasattr(model.config, "attention_probs_dropout_prob"):
                model.config.attention_probs_dropout_prob = float(dropout)

        return model, preprocess_info

    if model_name == "r3d18_fallback":
        model = R3D18Fallback(num_classes=num_classes)
        return model, preprocess_info

    raise ValueError(f"Unknown model_name: {model_name}")
