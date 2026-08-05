"""Special token definitions shared by the tokenizer, training and
fine-tuning pipelines.

Keeping these in one place means the tokenizer trainer, the pretraining
dataset loader, and the instruction-tuning formatter can never disagree
about which strings are "special" and what their reserved ids are.
"""

from __future__ import annotations

# Reserved control tokens. Order matters: SentencePiece assigns ids in the
# order pieces are declared, so this list *is* the id assignment for
# ids [0, len(SPECIAL_TOKENS)).
PAD = "<pad>"
UNK = "<unk>"
BOS = "<s>"
EOS = "</s>"

# Instruction-tuning / chat structure tokens. These let the fine-tuning
# format (see finetuning/format.py) mark turn boundaries explicitly instead
# of relying on natural-language delimiters that could appear in user text.
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END_TURN = "<|end|>"

SPECIAL_TOKENS: list[str] = [
    PAD,
    UNK,
    BOS,
    EOS,
    SYSTEM,
    USER,
    ASSISTANT,
    END_TURN,
]

# Fixed ids for the tokens the model/training code references directly.
# SentencePiece guarantees pieces declared via --control_symbols /
# --user_defined_symbols keep the order they were declared in, immediately
# after the three built-in ids SentencePiece always reserves (unk/bos/eos
# by default), so we instead declare all four of pad/unk/bos/eos ourselves
# for full control over ids (see tokenizer/trainer.py).
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SYSTEM_ID = 4
USER_ID = 5
ASSISTANT_ID = 6
END_TURN_ID = 7
