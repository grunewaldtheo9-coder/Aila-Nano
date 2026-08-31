"""Token-stream dataset for pretraining.

Design decision: pack the whole tokenized corpus into one flat binary file
of uint16 token ids (vocab_size <= 8192 comfortably fits in 16 bits) and
memory-map it (nanoGPT-style). This avoids ever materializing the full
dataset in RAM, makes random access O(1), and keeps the dataloader trivial
— a training example is just a contiguous slice of the memmap starting at
a random offset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

TOKEN_DTYPE = np.uint16


def write_token_bin(token_ids: list[int] | np.ndarray, path: str) -> None:
    """Write a sequence of token ids to a flat uint16 binary file."""
    arr = np.asarray(token_ids, dtype=TOKEN_DTYPE)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


class TokenBinDataset(Dataset):
    """Randomly samples fixed-length (input, target) windows from a flat
    token binary file, memory-mapped so it never has to fit in RAM.

    Each item is a `(seq_len)` input chunk and its next-token target
    (i.e. target[i] = input[i+1]), matching the standard causal LM setup.
    """

    def __init__(self, bin_path: str, seq_len: int, samples_per_epoch: int | None = None):
        self.bin_path = bin_path
        self.seq_len = seq_len
        self._mm = np.memmap(bin_path, dtype=TOKEN_DTYPE, mode="r")
        if len(self._mm) < seq_len + 1:
            raise ValueError(
                f"Token file {bin_path} has only {len(self._mm)} tokens, "
                f"need at least seq_len + 1 = {seq_len + 1}"
            )
        # An "epoch" is a virtual notion here since sampling is with
        # replacement at random offsets; default to covering the corpus
        # roughly once.
        self.samples_per_epoch = samples_per_epoch or max(
            1, len(self._mm) // seq_len
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self._mm) - self.seq_len - 1
        start = int(torch.randint(0, max_start + 1, (1,)).item())
        chunk = np.array(self._mm[start : start + self.seq_len + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

    @property
    def num_tokens(self) -> int:
        return len(self._mm)


def token_bin_stats(bin_path: str) -> dict:
    mm = np.memmap(bin_path, dtype=TOKEN_DTYPE, mode="r")
    return {"num_tokens": int(len(mm)), "path": bin_path}


def corpus_fingerprint(bin_path: str, chunk_bytes: int = 8 << 20) -> dict:
    """A reproducible identity for a tokenized corpus: a content hash plus
    the token count, computed by streaming the file so it never loads the
    whole corpus into RAM. Two `.bin` files with the same tokens produce
    the same `sha256`, so a checkpoint can record exactly which dataset it
    was trained on (spec: dataset versioning / reproducibility)."""
    import hashlib

    h = hashlib.sha256()
    size = 0
    with open(bin_path, "rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            h.update(block)
            size += len(block)
    return {
        "sha256": h.hexdigest(),
        "num_tokens": size // np.dtype(TOKEN_DTYPE).itemsize,
        "bytes": size,
        "path": str(bin_path),
    }
