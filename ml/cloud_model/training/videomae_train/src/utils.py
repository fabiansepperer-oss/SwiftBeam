#!/usr/bin/env python3
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Union[str, Path]) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def now_ts() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = v
    return out


def load_config(main_cfg_path: str, override_cfg_path: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_yaml(main_cfg_path)

    # Optional include for paths
    if "paths" in cfg and isinstance(cfg["paths"], str):
        paths_cfg = load_yaml(cfg["paths"])
        cfg = merge_dicts(paths_cfg, cfg)
        cfg.pop("paths", None)

    if override_cfg_path:
        over = load_yaml(override_cfg_path)
        cfg = merge_dicts(cfg, over)

    return cfg


@dataclass
class AverageMeter:
    name: str
    value: float = 0.0
    count: int = 0

    def update(self, v: float, n: int = 1) -> None:
        self.value += v * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.value / self.count


def save_json(path: Union[str, Path], obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def append_jsonl(path: Union[str, Path], obj: Any) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def save_checkpoint(
    ckpt_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    best_val: float,
    cfg: Dict[str, Any],
) -> None:
    payload = {
        "epoch": epoch,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, ckpt_path)
