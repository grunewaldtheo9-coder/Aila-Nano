# datasets/instruction — general instruction-following data

**Purpose:** teach the (already language-capable) model to follow
instructions and answer questions — general, task-shaped, NOT Aila-specific.
Applied as a fine-tuning stage AFTER general pretraining.

Kept separate from `datasets/pretrain/` (which must stay general language)
and `datasets/aila/` (identity/personality). See spec objective 5 and the
training-stage pipeline in `docs/AILA_NANO_50M.md`.
