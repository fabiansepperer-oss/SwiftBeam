#!/usr/bin/env bash
set -e

# Base train
python3 -u src/train.py --config configs/base_T8_192.yaml --test_after

# Stage A (head only) on top of base config
python3 -u src/train.py --config configs/base_T8_192.yaml --override configs/stageA_head_only.yaml --test_after

# Stage B (unfreeze last 2 blocks) on top of base config
python3 -u src/train.py --config configs/base_T8_192.yaml --override configs/stageB_unfreeze_last2.yaml --test_after
