# datasets/pretrain — general language/knowledge corpora

**Purpose:** teach the model *language* — grammar, vocabulary, general
knowledge, text structure, English + Portuguese. This stage must NOT be
contaminated with Aila identity/personality or instruction-formatting data
(those live in `datasets/instruction/` and `datasets/aila/`). Keeping them
separate is what lets us measure general language capability independently
of Aila-specific behaviour (spec objective 5).

Each corpus version has a manifest at `<version>/manifest.json` recording
sources, licenses/provenance, content hash, token counts, language
distribution, and the filter config — so a checkpoint can name exactly the
data that produced it (see `datasets/scripts/dataset_manifest.py`).

Current versions:
- `aila_pretrain_v1_tinystories` — TinyStories (CDLA-Sharing-1.0), English.
  The corpus the shipped 20M and current 50M were trained on.

Build/refresh a corpus with the pipeline scripts in `datasets/scripts/`
(`clean_text.py`, `dedupe.py`, `langid.py`, `prepare_pretrain.py`) and
record it with `dataset_manifest.py`.
