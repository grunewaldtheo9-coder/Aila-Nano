"""Trains a SentencePiece BPE tokenizer for Aila Nano.

Design decision: SentencePiece (BPE mode) over a hand-rolled tokenizer.
SentencePiece is battle-tested, language-agnostic (no pre-tokenization
assumptions), trains in seconds even on CPU, and produces a single
self-contained `.model` file that's trivial to version and ship. Byte
fallback is enabled so the tokenizer can represent *any* UTF-8 input
(emoji, code, non-English text) without ever producing <unk>, which keeps
the vocabulary small while staying lossless — important for a model this
small, where every embedding row is a meaningful fraction of the
parameter budget.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import sentencepiece as spm

from tokenizer.special_tokens import SPECIAL_TOKENS

logger = logging.getLogger(__name__)

DEFAULT_VOCAB_SIZE = 8192


def train_tokenizer(
    input_files: list[str] | str,
    model_prefix: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    max_sentence_length: int = 8192,
    num_threads: int = 4,
    input_sentence_size: int = 0,
    shuffle_input_sentence: bool = True,
) -> str:
    """Train a SentencePiece tokenizer and write `<model_prefix>.model/.vocab`.

    Args:
        input_files: one path, or a list of paths, to plain-text training
            corpora (one document/sentence per line works best).
        model_prefix: output path prefix, e.g. "tokenizer/artifacts/aila_nano".
        vocab_size: total vocabulary size, including the special tokens.
        model_type: "bpe" (default) or "unigram".
        character_coverage: fraction of characters in the corpus that must
            be covered by the vocabulary; 0.9995 is the SentencePiece
            recommendation for languages with large character sets, and is
            safe for English + code + light multilingual text too.
        max_sentence_length: max bytes per training line.
        num_threads: SentencePiece trainer thread count.
        input_sentence_size: if >0, subsample the corpus to this many lines
            before training (SentencePiece keeps the whole corpus in RAM
            otherwise). 0 means "use everything".
        shuffle_input_sentence: shuffle before subsampling.

    Returns:
        Path to the trained `.model` file.
    """
    if isinstance(input_files, str):
        input_files = [input_files]

    for f in input_files:
        if not Path(f).exists():
            raise FileNotFoundError(f"Tokenizer training input not found: {f}")

    out_dir = Path(model_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError(
            f"vocab_size ({vocab_size}) must exceed the number of reserved "
            f"special tokens ({len(SPECIAL_TOKENS)})"
        )

    # We declare pad/unk/bos/eos explicitly (rather than relying on
    # SentencePiece's defaults) so their ids are guaranteed stable and equal
    # to the ids in tokenizer/special_tokens.py, and we add the chat-format
    # tokens as user-defined symbols so they're never split into subwords.
    pad, unk, bos, eos = SPECIAL_TOKENS[0:4]
    extra_symbols = SPECIAL_TOKENS[4:]

    logger.info(
        "Training SentencePiece tokenizer: vocab_size=%d model_type=%s inputs=%s",
        vocab_size,
        model_type,
        input_files,
    )

    spm.SentencePieceTrainer.train(
        input=",".join(input_files),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        byte_fallback=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece=pad,
        unk_piece=unk,
        bos_piece=bos,
        eos_piece=eos,
        user_defined_symbols=extra_symbols,
        max_sentence_length=max_sentence_length,
        num_threads=num_threads,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=shuffle_input_sentence,
        normalization_rule_name="nfkc",
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
    )

    model_path = f"{model_prefix}.model"
    logger.info("Tokenizer written to %s", model_path)
    return model_path


def train_tokenizer_from_iterator(
    text_iterator,
    model_prefix: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    **kwargs,
) -> str:
    """Same as `train_tokenizer` but streams training text from any Python
    iterator of strings, avoiding an intermediate file for small/ad-hoc
    corpora (e.g. in tests or notebooks).
    """
    out_dir = Path(model_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pad, unk, bos, eos = SPECIAL_TOKENS[0:4]
    extra_symbols = SPECIAL_TOKENS[4:]

    model_buffer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(text_iterator),
        model_writer=model_buffer,
        vocab_size=vocab_size,
        model_type=kwargs.pop("model_type", "bpe"),
        character_coverage=kwargs.pop("character_coverage", 0.9995),
        byte_fallback=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece=pad,
        unk_piece=unk,
        bos_piece=bos,
        eos_piece=eos,
        user_defined_symbols=extra_symbols,
        normalization_rule_name="nfkc",
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        **kwargs,
    )
    model_path = f"{model_prefix}.model"
    with open(model_path, "wb") as f:
        f.write(model_buffer.getvalue())
    return model_path
