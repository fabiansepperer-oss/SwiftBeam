#!/usr/bin/env bash
set -euo pipefail

# ---- CUDA preflight (fails fast if CUDA hangs) ----
cuda_preflight () {
  echo "[probe] Checking CUDA (timeout 8s) ..."
  timeout 8 python3 - <<'PY'
import torch, time
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available")

d = torch.device("cuda:0")
print("device:", torch.cuda.get_device_name(0))

# tiny kernel + sync (catches many 'hang' states)
x = torch.randn(1024, 1024, device=d)
y = (x @ x).sum()
torch.cuda.synchronize()
print("ok: kernel+sync")

# AMP sanity (optional)
with torch.cuda.amp.autocast(True):
    z = (x * 1.1).mean()
torch.cuda.synchronize()
print("ok: amp")

print("ok: preflight done")
PY
}

cuda_preflight

# Base train
python3 -u src/train.py --config configs/base_T16_192.yaml --test_after

cuda_preflight

# Stage A (head only) on top of base config
python3 -u src/train.py --config configs/base_T16_192.yaml --override configs/stageA_head_only.yaml --test_after

cuda_preflight

# Stage B (unfreeze last 2 blocks) on top of base config
python3 -u src/train.py --config configs/base_T16_192.yaml --override configs/stageB_unfreeze_last2.yaml --test_after
