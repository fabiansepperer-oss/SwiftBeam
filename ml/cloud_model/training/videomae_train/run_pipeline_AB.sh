#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   STAGEA_CFG=configs/stageA_head_only.yaml STAGEB_CFG=configs/stageB_unfreeze_last2.yaml ./run_pipeline_AB.sh
#   SKIP_STAGEA=1 ./run_pipeline_AB.sh
#   PYTHON=python3 ./run_pipeline_AB.sh

PYTHON="${PYTHON:-python3}"
STAGEA_CFG="${STAGEA_CFG:-configs/stageA_head_only.yaml}"
STAGEB_CFG="${STAGEB_CFG:-configs/stageB_unfreeze_last2.yaml}"
BASE_CFG="configs/base_T16_192.yaml"

resolve_stagea_dir() {
  local outputs_root exp_name

  if mapfile -t _info < <(BASE_CFG="$BASE_CFG" STAGEA_CFG="$STAGEA_CFG" "$PYTHON" - <<'PY'
import os
import sys

sys.path.insert(0, "src")
from utils import load_config

cfg = load_config(os.environ["BASE_CFG"], os.environ["STAGEA_CFG"])
print(cfg.get("outputs_root", "outputs"))
print(cfg.get("exp_name", "exp_stageA_head_only"))
PY
  ); then
    outputs_root="${_info[0]}"
    exp_name="${_info[1]}"
    if [[ -n "$outputs_root" && -n "$exp_name" ]]; then
      echo "${outputs_root%/}/${exp_name}"
      return
    fi
  fi

  local newest
  newest=$(ls -td outputs/exp_* 2>/dev/null | head -n1 || true)
  if [[ -n "$newest" ]]; then
    echo "$newest"
    return
  fi

  echo ""
}

if [[ "${SKIP_STAGEA:-0}" != "1" ]]; then
  "$PYTHON" -u src/train.py --config "$BASE_CFG" --override "$STAGEA_CFG"
fi

stagea_dir="$(resolve_stagea_dir)"
if [[ -z "$stagea_dir" ]]; then
  echo "[ERROR] Could not resolve Stage A outputs directory."
  exit 1
fi

best_ckpt="$stagea_dir/checkpoints/best.pt"
last_ckpt="$stagea_dir/checkpoints/last.pt"

if [[ -f "$best_ckpt" ]]; then
  init_ckpt="$best_ckpt"
elif [[ -f "$last_ckpt" ]]; then
  init_ckpt="$last_ckpt"
else
  echo "[ERROR] No Stage A checkpoint found in $stagea_dir/checkpoints"
  exit 1
fi

echo "[INFO] Using init checkpoint: $init_ckpt"

"$PYTHON" -u src/train.py --config "$BASE_CFG" --override "$STAGEB_CFG" --init_from "$init_ckpt"
