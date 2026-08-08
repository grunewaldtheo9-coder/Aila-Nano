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
import glob
import sys

target = int(sys.argv[1])

# Check the FURTHEST-ALONG checkpoint, not best.pt. best.pt tracks the
# lowest validation loss, which is routinely an *earlier* step than the
# final one (this run: best.pt=650, final=700). An earlier version of
# this script compared best.pt's step against the target, concluded
# training had stopped short, and relaunched a finished run — burning
# its whole retry budget and then refusing to fine-tune at all.
paths = glob.glob("checkpoints/pretrain_20m/*.pt")
if not paths:
    sys.exit(1)

import torch

best = 0
for p in paths:
    try:
        best = max(best, torch.load(p, map_location="cpu", weights_only=False).get("step", 0))
    except Exception:
        continue  # a half-written checkpoint shouldn't crash the gate
sys.exit(0 if best >= target else 1)
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
