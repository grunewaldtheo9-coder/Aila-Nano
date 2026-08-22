"""Multi-turn chat dataset (the `messages` JSONL schema).

Parallels finetuning/dataset.py's InstructionDataset, but consumes whole
conversations via finetuning/chat_format.py. Invalid records are skipped
with a warning (validated up front) rather than crashing training, and the
same next-token shift the instruction dataset applies is applied here so
both training paths mean the same thing by `targets`. Batches use the same
`collate_instruction_batch`, so the existing training loop can consume a
ChatDataset unchanged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

from finetuning.chat_format import (
    ChatConversation,
    encode_conversation,
    validate_conversation,
)
from tokenizer.tokenizer import AilaTokenizer

logger = logging.getLogger(__name__)


class ChatDataset(Dataset):
    def __init__(
        self,
        jsonl_paths: str | list[str],
        tokenizer: AilaTokenizer,
        max_seq_len: int = 1024,
    ):
        if isinstance(jsonl_paths, str):
            jsonl_paths = [jsonl_paths]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self._encoded: list[tuple[list[int], list[int]]] = []
        n_invalid = n_skipped = 0
        for path in jsonl_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Chat dataset not found: {path}")
            for line_no, line in enumerate(
                Path(path).read_text(encoding="utf-8").splitlines(), 1
            ):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    n_invalid += 1
                    logger.warning("%s:%d: invalid JSON, skipped", path, line_no)
                    continue
                reason = validate_conversation(record)
                if reason is not None:
                    n_invalid += 1
                    logger.warning("%s:%d: %s, skipped", path, line_no, reason)
                    continue
                result = encode_conversation(
                    ChatConversation.from_dict(record), tokenizer, max_len=max_seq_len
                )
                if result is None:
                    n_skipped += 1
                    continue
                self._encoded.append(result)

        if n_invalid:
            logger.warning("Skipped %d invalid conversation record(s).", n_invalid)
        if n_skipped:
            logger.warning(
                "Skipped %d conversation(s) too long to fit max_seq_len=%d.",
                n_skipped,
                max_seq_len,
            )
        if not self._encoded:
            raise ValueError(f"No usable conversations loaded from {jsonl_paths}")

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids, labels = self._encoded[idx]
        # Same next-token shift as InstructionDataset — the model's forward
        # does no internal shift, so labels must be moved one position left.
        return {
            "input_ids": torch.tensor(ids[:-1], dtype=torch.long),
            "labels": torch.tensor(labels[1:], dtype=torch.long),
        }
