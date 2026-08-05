# Model Card: Aila Nano

## Summary

| | |
|---|---|
| Developer | Aila Company Solutions |
| Model type | Decoder-only transformer (causal language model) |
| Parameters | 10,877,184 (~10.88M), default config |
| Context length | 512 tokens |
| Vocabulary | 8,192 tokens (SentencePiece BPE, byte-fallback) |
| Precision | fp32 training-compatible; bf16/fp16 autocast supported |
| License | Apache 2.0 (code); see `datasets/README.md` for data licenses |
| Languages | Primarily English (byte-fallback tokenizer handles any UTF-8 input without loss) |

## Aila Company Solutions

Aila Nano is designed, trained, and maintained by **Aila Company
Solutions**, founded by **Theo Grunewald Hames** and **Guilherme
Grunewald Benkendorf**. Aila Nano is intended as the first model in a
future family of language models built by Aila Company Solutions — see
`datasets/aila_knowledge/aila_company.jsonl` for the instruction data
that teaches the model this fact about itself.

## Intended use

- Local, offline text generation, question answering, and conversation
  on commodity hardware (including CPU-only machines).
- A base for fine-tuning on new instruction data (`finetuning/`).
- A teaching/reference implementation: small enough to read, train, and
  modify end-to-end without specialized infrastructure.
- Semantic search / retrieval over small document collections, using the
  model's own embeddings (`vectordb/`).

## Out of scope / limitations

- **Scale-appropriate expectations.** At ~10.9M parameters, Aila Nano has
  roughly 1/100th to 1/1000th the parameters of contemporary
  general-purpose chat models. It will not match their world knowledge,
  reasoning depth, or robustness to adversarial prompts. It is well
  suited to narrow, well-specified tasks and to being fine-tuned toward a
  specific domain; it is not a general-purpose knowledge base.
- **Factuality.** Like any language model, Aila Nano can produce
  plausible-sounding but incorrect statements ("hallucination"), and this
  risk is *higher*, not lower, at small scale. Don't use raw model output
  for high-stakes factual claims without verification. The memory/RAG
  system (`vectordb/`, `memory/`) is provided specifically to ground
  responses in retrieved, sourced text rather than relying on parametric
  knowledge alone.
- **Safety.** No dedicated safety/RLHF alignment pass is included in this
  repository's training pipeline. If you fine-tune Aila Nano for a
  user-facing product, add your own safety fine-tuning/evaluation
  appropriate to your use case before deployment.
- **Language coverage.** The default tokenizer and bundled datasets are
  English-centric. The tokenizer itself (byte-fallback SentencePiece) can
  represent any UTF-8 text, but a model pretrained only on English data
  will not perform well generating other languages without additional
  multilingual training data.

## Training data

See [`datasets/README.md`](../datasets/README.md) for the full list of
sources and their licenses. In summary: TinyStories (CDLA-Sharing-1.0)
and Wikitext-103 (CC BY-SA 3.0) for pretraining; a small original
`aila_knowledge` instruction set plus (optionally) additional
permissively-licensed instruction data for fine-tuning.

## Evaluation

`training/trainer.py` reports validation cross-entropy loss and
perplexity (`exp(loss)`) at each `eval_interval`, logged to TensorBoard
(`tensorboard --logdir runs/`). No standardized third-party benchmark
suite is run automatically as part of this repository; if you need
benchmark numbers for a specific downstream use case, evaluate the
fine-tuned checkpoint against that use case directly.

## Reproducibility

Every architectural and training hyperparameter lives in a checked-in
config (`configs/model/nano_10m.yaml`, `configs/training/*.yaml`) or the
`GPTConfig`/`TrainingConfig`/`FinetuneConfig` dataclasses that back them.
`scripts/count_params.py` reproduces the exact parameter count reported
above from `model/config.py::nano_10m()`.
