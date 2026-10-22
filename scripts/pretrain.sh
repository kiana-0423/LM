#!/usr/bin/env bash
# Run explicitly from the repository root. No training happens when editing this file.
set -euo pipefail

"${PYTHON:-.venv/bin/python}" -m hanlm.cli train \
  --stage pretrain --method full \
  --config configs/100m.json --tokenizer artifacts/zh/tokenizer.json \
  --train artifacts/pretrain/train.bin --val artifacts/pretrain/val.bin \
  --output runs/100m-pretrain --device cpu \
  --seq-len 512 --batch-size 128 --grad-accum 1 \
  --optimizer adamw --lr 3e-4 --steps 120000 \
  --warmup-steps 50 --lr-scheduler cosine \
  --eval-every 100 --save-every 100 --gradient-checkpointing \
  "$@"
