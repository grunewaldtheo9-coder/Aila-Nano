# Training Aila Nano on a cloud GPU

Everything in this project trains on CPU today. That works — Aila Nano
Both model sizes were trained that way — but it caps how much data the
model can see. This guide is for the next, bigger run.

## Why bother: the actual numbers from this project

Measured on the CPU machine used for `nano_20m` (4 vCPU, 15 GB RAM), the
~19.8M-parameter model at `batch=16, seq=256`:

| | CPU (measured here) | One mid-range GPU (typical) |
|---|---|---|
| Throughput | ~1,300 tokens/sec | ~150,000–400,000 tokens/sec |
| 700 steps (11.5M tokens) | ~2.5 hours | ~1–2 minutes |
| One full epoch (17.4M tokens) | ~3.7 hours | ~2 minutes |
| 10 epochs over 174M tokens | ~37 hours | ~20 minutes |

The point isn't speed for its own sake. **A ~20M model needs far more
data than 11.5M tokens to be good** — rules of thumb put the
compute-optimal figure in the hundreds of millions to low billions of
tokens. On CPU that's weeks; on one GPU it's an afternoon. That is the
single biggest quality lever left in this project, much bigger than
adding parameters.

The CPU bottleneck is specifically attention: PyTorch has no
flash-attention kernel on CPU, so the math fallback materializes full
attention matrices. This is also why the real runs train at `seq=256`
instead of the 512 spec context — halving sequence length roughly
doubles CPU throughput.

## What to rent

Aila Nano is small; you do not need an expensive GPU. Any of these
fit the whole model, optimizer state, and a healthy batch in memory:

| GPU | VRAM | Rough cost/hr | Good for |
|---|---|---|---|
| RTX 3090 / 4090 | 24 GB | $0.20–0.40 | Best value for this size |
| A10G / L4 | 24 GB | $0.50–0.75 | Widely available |
| A100 40GB | 40 GB | $1.00–2.00 | Overkill here |

Providers, cheapest-first in practice: **Vast.ai**, **RunPod**,
**Lambda Labs**, then the big clouds (AWS/GCP/Azure — more setup, more
paperwork). **Google Colab** is free but disconnects, which is fine for
experiments and bad for long runs.

Budget check: a serious 5-hour run on a 4090 costs roughly **$1–2**.

> ⚠️ Billing is on you. Machines bill per second **while running, even
> when idle**. Destroy the instance when finished — not just "stop" it,
> unless you intend to keep paying for its disk.

## Step by step

### 1. Rent the machine

Pick a provider above, choose a GPU, and select a **PyTorch** or **CUDA**
template image. Note the SSH command it gives you.

### 2. Connect and get the code

```bash
ssh root@<ip> -p <port>          # exact command comes from the provider

git clone https://github.com/grunewaldtheo9-coder/Aila-Nano.git
cd Aila-Nano
git checkout claude/aila-nano-slm-h8nzs0
```

### 3. Install (GPU build of PyTorch, not the CPU one)

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Confirm the GPU is actually visible before training anything:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If that prints `False`, stop and fix it — otherwise you will pay GPU
prices to train on CPU.

### 4. Get the training data

The token shards aren't in Git (too large). Rebuild them on the machine:

```bash
pip install -r requirements-datasets.txt
python datasets/scripts/download_pretrain_data.py
python datasets/scripts/prepare_pretrain_data.py \
  --tokenizer tokenizer/artifacts/aila_nano.model
```

### 5. Make a GPU training config

Copy `configs/training/pretrain_20m.yaml` and change these lines — the
CPU config is deliberately tiny and would waste the GPU:

```yaml
batch_size: 64          # was 16 — GPUs like big batches
grad_accum_steps: 4     # effective batch 64 x 4 x 512 = 131k tokens/step
max_steps: 20000        # was 700 — this is the whole point
device: cuda            # was cpu
amp: true               # was false — mixed precision, ~2x faster on GPU
eval_interval: 500
checkpoint_dir: checkpoints/pretrain_20m_gpu
```

Also copy `configs/model/nano20m_real_run.yaml` and set
`max_seq_len: 512` — the full spec context, which is affordable on GPU.

If you hit out-of-memory, halve `batch_size` and double
`grad_accum_steps` to keep the effective batch identical.

### 6. Train (detached, so it survives disconnection)

```bash
# tmux keeps it alive if your SSH connection drops
tmux new -s train

python -m training.train \
  --model-config configs/model/nano20m_gpu.yaml \
  --train-config configs/training/pretrain_20m_gpu.yaml

# Ctrl+B then D to detach; `tmux attach -t train` to come back
```

Better still, use the supervisor already in this repo — it auto-resumes
on a crash and refuses to fine-tune a half-trained model:

```bash
# edit the TARGET_STEPS and config paths at the top first
./scripts/supervise_training.sh
```

### 7. Fine-tune and check quality

```bash
python -m finetuning.finetune \
  --init-checkpoint checkpoints/pretrain_20m_gpu/best.pt \
  --tokenizer tokenizer/artifacts/aila_nano.model \
  --data datasets/aila_knowledge/aila_company.jsonl \
         datasets/sample/finetune_sample.jsonl \
         datasets/aila_knowledge/portuguese_basic.jsonl \
  --config configs/training/finetune_20m.yaml

python scripts/benchmark_model.py --checkpoint checkpoints/finetune_20m/best.pt
```

Compare the benchmark against the current numbers before promoting
anything — **loss going down is not proof the chat model got better.**
This project has a scar from exactly that: a run with a beautiful loss
curve generated pure garbage, because `input_ids` and `labels` weren't
shifted and the model was learning to echo instead of predict. Always
look at real generated text, not just the metric.

### 8. Get the checkpoint off the machine, then destroy it

```bash
# from your own computer
scp -P <port> root@<ip>:/root/Aila-Nano/checkpoints/finetune_20m/best.pt .
```

Then **destroy the instance** in the provider's console. This is the step
people forget and it is the one that costs money.

## If you want more than a faster repeat

Once you have GPU throughput, the highest-value changes are about data,
not hardware:

1. **More and better pretraining data.** The current corpus is
   TinyStories (~17.4M tokens, simple English children's stories). That
   ceiling is why the model sounds childlike. Consider FineWeb-Edu or
   similar for real general text.
2. **Portuguese in *pretraining*, not just fine-tuning.** The current
   Portuguese support is a thin instruction layer over an English-only
   base — the honest limit documented in
   [MODEL_CARD.md](MODEL_CARD.md). A bilingual corpus (and retraining
   the tokenizer on it) is what would make Portuguese genuinely work.
3. **Longer context.** `seq=512` costs almost nothing on GPU and the
   architecture already supports RoPE scaling beyond it.
4. **More instruction data.** ~100 examples is very few; thousands is
   where instruction-following starts to become reliable.
