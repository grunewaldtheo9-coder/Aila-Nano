"""JSONL instruction-tuning dataset."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

from finetuning.format import IGNORE_INDEX, InstructionExample, encode_example
from tokenizer.tokenizer import AilaTokenizer

logger = logging.getLogger(__name__)


class InstructionDataset(Dataset):
    def __init__(
        self,
        jsonl_paths: str | list[str],
        tokenizer: AilaTokenizer,
        max_seq_len: int = 512,
    ):
        if isinstance(jsonl_paths, str):
            jsonl_paths = [jsonl_paths]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        raw_examples: list[InstructionExample] = []
        for path in jsonl_paths:
            raw_examples.extend(_load_jsonl(path))
        if not raw_examples:
            raise ValueError(f"No examples loaded from {jsonl_paths}")

        # Pre-encode every example up front so we can drop the ones whose
        # assistant response can't fit in max_seq_len at all — see
        # finetuning/format.py::encode_example. Encoding them lazily in
        # __getitem__ would let a bad example silently produce a
        # zero-trainable-token (nan-loss) batch deep inside training.
        self._encoded: list[tuple[list[int], list[int]]] = []
        n_skipped = 0
        for example in raw_examples:
            result = encode_example(example, tokenizer, max_len=max_seq_len)
            if result is None:
                n_skipped += 1
                continue
            self._encoded.append(result)

        if n_skipped:
            logger.warning(
                "Skipped %d/%d instruction example(s) whose response alone exceeds "
                "max_seq_len=%d and cannot be trained on.",
                n_skipped,
                len(raw_examples),
                max_seq_len,
            )
        if not self._encoded:
            raise ValueError(
                f"All {len(raw_examples)} example(s) exceed max_seq_len={max_seq_len}; "
                f"nothing left to train on."
            )

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids, labels = self._encoded[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _load_jsonl(path: str) -> list[InstructionExample]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Instruction dataset not found: {path}")
    examples = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON — {e}") from e
            if "instruction" not in d or "output" not in d:
                raise ValueError(
                    f"{path}:{line_no}: example must have 'instruction' and 'output' fields"
                )
            examples.append(InstructionExample.from_dict(d))
    return examples


def collate_instruction_batch(
    batch: list[dict[str, torch.Tensor]], pad_id: int
) -> dict[str, torch.Tensor]:
    """Right-pad a batch of variable-length (input_ids, labels) to the
    batch's max length. Padding positions get label IGNORE_INDEX so they
    never contribute to the loss; causal attention means they also never
    influence any real token's prediction.
    """
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["input_ids"].shape[0]
        input_ids[i, :n] = item["input_ids"]
        labels[i, :n] = item["labels"]
    return {"input_ids": input_ids, "labels": labels}
