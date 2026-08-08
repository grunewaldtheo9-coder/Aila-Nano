#!/bin/bash
# Supervisor for the Aila Nano 2.0 cloud training run.
#
# Two jobs:
#   1. If pretraining dies before reaching max_steps (container hiccup,
#      OOM kill, transient crash), resume it from the last checkpoint —
#      checkpoints carry full model+optimizer+step state, so a resume
#      loses at most one checkpoint interval (50 steps).
#   2. Only launch the fine-tune once pretraining has ACTUALLY reached
#      completion. The previous chain script only checked that best.pt
#      existed, which is true even mid-run — a crash at step 350 would
#      have silently fine-tuned a half-trained model.
cd /home/user/Aila-Nano

TARGET_STEPS=700
MAX_RESUMES=5
resumes=0

reached_target() {
  .venv/bin/python - "$TARGET_STEPS" <<'PY' 2>/dev/null
import sys
from pathlib import Path
target = int(sys.argv[1])
ck = Path("checkpoints/pretrain_20m/best.pt")
if not ck.exists():
    sys.exit(1)
import torch
step = torch.load(ck, map_location="cpu", weights_only=False).get("step", 0)
sys.exit(0 if step >= target else 1)
PY
}

while true; do
  # Wait while a pretraining process is alive.
  while pgrep -f "training.train.*pretrain_20m" > /dev/null; do
    sleep 30
  done

  if reached_target; then
    echo "$(date -Is) pretraining reached step >= $TARGET_STEPS"
    break
  fi

  if [ "$resumes" -ge "$MAX_RESUMES" ]; then
    echo "$(date -Is) ERROR: pretraining stopped short and hit the resume limit ($MAX_RESUMES). Not fine-tuning a half-trained model."
    exit 1
  fi

  resumes=$((resumes + 1))
  echo "$(date -Is) pretraining stopped before step $TARGET_STEPS — resuming (attempt $resumes/$MAX_RESUMES)"
  nohup .venv/bin/python -u -m training.train \
    --model-config configs/model/nano20m_real_run.yaml \
    --train-config configs/training/pretrain_20m.yaml \
    --resume >> /tmp/pretrain_20m.log 2>&1 &
  sleep 60
done

echo "$(date -Is) launching fine-tune"
exec .venv/bin/python -u -m finetuning.finetune \
  --init-checkpoint checkpoints/pretrain_20m/best.pt \
  --tokenizer tokenizer/artifacts/aila_nano.model \
  --data datasets/aila_knowledge/aila_company.jsonl datasets/sample/finetune_sample.jsonl datasets/aila_knowledge/portuguese_basic.jsonl \
  --config configs/training/finetune_20m.yaml
