#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Jetson-safe defaults
# -----------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CUDA_PROBE_TIMEOUT="${CUDA_PROBE_TIMEOUT:-5}"   # seconds
PYTHON="${PYTHON:-python3}"

# Run from repo root (videomae_train/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

STEPS="${STEPS:-100}"           # override like: STEPS=200 ./benchmark.sh
RUN_STAGES="${RUN_STAGES:-0}"   # 1 to also benchmark stageA/stageB overrides

cleanup() {
  # Best-effort: kill leftover python processes started from this repo
  pkill -f "python.*${ROOT_DIR//\//\\/}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Repo: $ROOT_DIR"
echo "Python: $PYTHON"
echo "Benchmark steps: $STEPS"
echo "RUN_STAGES: $RUN_STAGES"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_PROBE_TIMEOUT: ${CUDA_PROBE_TIMEOUT}s"
echo

cuda_probe () {
  # Do NOT call torch.cuda.is_available() here (can hang on Jetson).
  # Use device_count() behind a timeout to prevent indefinite hangs.
  echo "[probe] Checking CUDA (timeout ${CUDA_PROBE_TIMEOUT}s) ..."
  set +e
  timeout "${CUDA_PROBE_TIMEOUT}" bash -lc \
    "$PYTHON - << 'EOF'
import torch
print('torch:', torch.__version__)
print('torch.version.cuda:', torch.version.cuda)
print('device_count:', torch.cuda.device_count())
EOF" >/tmp/bench_cuda_probe.log 2>&1
  rc=$?
  set -e

  if [[ $rc -eq 124 ]]; then
    echo "[probe] ERROR: CUDA probe timed out (likely CUDA/nvmap init hang). Aborting to keep node stable."
    echo "[probe] Last probe log:"
    sed -n '1,200p' /tmp/bench_cuda_probe.log || true
    exit 40
  elif [[ $rc -ne 0 ]]; then
    echo "[probe] ERROR: CUDA probe failed with exit code $rc. Aborting."
    echo "[probe] Last probe log:"
    sed -n '1,200p' /tmp/bench_cuda_probe.log || true
    exit 41
  fi

  # Print short probe output for visibility
  sed -n '1,50p' /tmp/bench_cuda_probe.log || true
  echo "[probe] OK"
  echo
}

run_one () {
  local cfg="$1"
  local override="${2:-}"
  local label="$3"

  echo "=== Benchmark: $label ==="

  # Safety: probe before each run to avoid long hangs inside training
  cuda_probe

  if [[ -n "$override" ]]; then
    $PYTHON -u src/train.py \
      --config "$cfg" \
      --override "$override" \
      --bench_steps "$STEPS" \
      --bench_only
  else
    $PYTHON -u src/train.py \
      --config "$cfg" \
      --bench_steps "$STEPS" \
      --bench_only
  fi

  echo "Saved JSON under: outputs/<exp_name>/logs/bench_${STEPS}steps.json"
  echo
}

# One probe up front (fast fail if CUDA is broken)
cuda_probe

# Base configs
run_one "configs/base_T8_192.yaml"  "" "base_T8_192"
run_one "configs/base_T16_192.yaml" "" "base_T16_192"

# Optional: stage overrides (only meaningful if your train.py supports --override configs)
if [[ "$RUN_STAGES" == "1" ]]; then
  run_one "configs/base_T8_192.yaml"  "configs/stageA_head_only.yaml"       "T8 stageA_head_only"
  run_one "configs/base_T8_192.yaml"  "configs/stageB_unfreeze_last2.yaml"  "T8 stageB_unfreeze_last2"

  run_one "configs/base_T16_192.yaml" "configs/stageA_head_only.yaml"       "T16 stageA_head_only"
  run_one "configs/base_T16_192.yaml" "configs/stageB_unfreeze_last2.yaml"  "T16 stageB_unfreeze_last2"
fi

echo "All benchmarks done."
