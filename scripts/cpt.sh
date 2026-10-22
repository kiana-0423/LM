#!/usr/bin/env bash
# Full-parameter CPT; sequence length and batch size inherit the pretraining setup.
set -euo pipefail

"${PYTHON:-.venv/bin/python}" -m hanlm.cli train \
  --stage cpt --method full \
  --init runs/100m-pretrain/best.pt --tokenizer artifacts/zh/tokenizer.json \
  --train artifacts/cpt/train.bin --val artifacts/cpt/val.bin \
  --output runs/100m-cpt --device cpu \
  --seq-len 512 --batch-size 128 --grad-accum 1 \
  --optimizer adamw --lr 1e-4 --steps 15000 \
  --warmup-steps 500 --lr-scheduler linear \
  --eval-every 100 --save-every 100 --gradient-checkpointing \
  "$@"
