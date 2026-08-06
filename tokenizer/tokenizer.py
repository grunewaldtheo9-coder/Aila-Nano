"""Aila Nano tokenizer: a thin, typed wrapper around a trained
SentencePiece model exposing the encode/decode/save/load contract the
rest of the project depends on.
"""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

from tokenizer.special_tokens import (
    ASSISTANT_ID,
    BOS_ID,
    END_TURN_ID,
    EOS_ID,
    PAD_ID,
    SYSTEM_ID,
    UNK_ID,
    USER_ID,
)


class AilaTokenizer:
    """Encode/decode text with a trained SentencePiece model.

    Example:
        >>> tok = AilaTokenizer.load("tokenizer/artifacts/aila_nano.model")
        >>> ids = tok.encode("Hello, Aila!", add_bos=True, add_eos=True)
        >>> tok.decode(ids)
        'Hello, Aila!'
    """

    pad_id = PAD_ID
    unk_id = UNK_ID
    bos_id = BOS_ID
    eos_id = EOS_ID
    system_id = SYSTEM_ID
    user_id = USER_ID
    assistant_id = ASSISTANT_ID
    end_turn_id = END_TURN_ID

    def __init__(self, sp_model: spm.SentencePieceProcessor, model_path: str | None = None):
        self._sp = sp_model
        self.model_path = model_path
        self._byte_fallback_ids: list[int] | None = None

    # -- construction -----------------------------------------------------

    @classmethod
    def load(cls, model_path: str | Path) -> AilaTokenizer:
        model_path = str(model_path)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Tokenizer model not found at '{model_path}'. Train one first with "
                f"`python -m tokenizer.train` or `scripts/train_tokenizer.py`."
            )
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        return cls(sp, model_path=model_path)

    def save(self, path: str | Path) -> None:
        """Copy the underlying `.model` file to `path`. Training already
        writes the model file to disk (see tokenizer/trainer.py); this
        exists so callers can treat the tokenizer object itself as
        serializable, e.g. when packaging a checkpoint bundle.
        """
        if self.model_path is None:
            raise RuntimeError("Tokenizer has no backing model_path to save from.")
        import shutil

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.model_path, path)

    # -- core API -----------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._sp.vocab_size()

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = self._sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(
        self,
        texts: list[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[list[int]]:
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            special = {
                self.pad_id,
                self.unk_id,
                self.bos_id,
                self.eos_id,
                self.system_id,
                self.user_id,
                self.assistant_id,
                self.end_turn_id,
            }
            ids = [i for i in ids if i not in special]
        return self._sp.decode(ids)

    def piece_to_id(self, piece: str) -> int:
        return self._sp.piece_to_id(piece)

    def id_to_piece(self, idx: int) -> str:
        return self._sp.id_to_piece(idx)

    @property
    def byte_fallback_ids(self) -> list[int]:
        """Ids of the raw single-byte fallback pieces (`<0x00>`..`<0xFF>`)
        SentencePiece adds when `byte_fallback=True` (see
        tokenizer/trainer.py) so any UTF-8 input is representable without
        ever emitting `<unk>`. These are meant purely as an encoding
        safety net for characters outside the learned vocabulary — normal
        English text never needs them, so they almost never appear as a
        *target* during training and their embeddings stay poorly
        calibrated. Left unchecked, generation can occasionally latch onto
        one as a locally high-probability but semantically meaningless
        choice; `model.generate`/`generate_stream`'s `suppress_token_ids`
        uses this list to rule them out by default.
        """
        if self._byte_fallback_ids is None:
            self._byte_fallback_ids = [
                i
                for i in range(self.vocab_size)
                if self.id_to_piece(i).startswith("<0x") and self.id_to_piece(i).endswith(">")
            ]
        return self._byte_fallback_ids

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"AilaTokenizer(vocab_size={self.vocab_size}, model_path={self.model_path!r})"
