"""Multi-turn chat format: validation, encoding (loss only on assistant
turns, EOS on the last one), the inference template, and that every
shipped conversational dataset record is well-formed.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from finetuning.chat_format import (
    ChatConversation,
    encode_conversation,
    format_chat_for_inference,
    validate_conversation,
)
from finetuning.format import IGNORE_INDEX

REPO_ROOT = Path(__file__).resolve().parent.parent
CONV_DIR = REPO_ROOT / "datasets" / "conversational"


# -- validation ---------------------------------------------------------------


def test_valid_conversation_passes():
    conv = {
        "messages": [
            {"role": "system", "content": "You are Aila Nano."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hey!"},
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "See you!"},
        ]
    }
    assert validate_conversation(conv) is None


@pytest.mark.parametrize(
    "conv",
    [
        {"messages": []},
        {"messages": [{"role": "user", "content": "Hi"}]},  # no assistant turn
        {"messages": [{"role": "assistant", "content": "Hey"}]},  # starts with assistant
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "user", "content": "?"}]},  # two users
        {"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": "Hey"}]},  # empty
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hey"}, {"role": "system", "content": "x"}]},  # system not first
    ],
)
def test_invalid_conversations_are_rejected(conv):
    assert validate_conversation(conv) is not None


# -- encoding -----------------------------------------------------------------


def test_encoding_trains_only_on_assistant_turns(tokenizer):
    conv = ChatConversation.from_dict(
        {
            "messages": [
                {"role": "system", "content": "You are Aila Nano."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hey there"},
            ]
        }
    )
    ids, labels = encode_conversation(conv, tokenizer)
    assert len(ids) == len(labels)
    # Some labels are trained (the assistant span), some are masked.
    assert any(l != IGNORE_INDEX for l in labels)
    assert any(l == IGNORE_INDEX for l in labels)
    # The last real token trained on is EOS, so the model learns to stop.
    assert ids[-1] == tokenizer.eos_id
    assert labels[-1] == tokenizer.eos_id


def test_multi_turn_trains_on_every_assistant_turn(tokenizer):
    conv = ChatConversation.from_dict(
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hey!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "Good!"},
            ]
        }
    )
    ids, labels = encode_conversation(conv, tokenizer)
    trained = [i for i, l in zip(ids, labels) if l != IGNORE_INDEX]
    # Both assistant replies contribute: the EOS appears once (last turn),
    # and end-of-turn tokens appear for each assistant turn.
    assert trained.count(tokenizer.eos_id) == 1
    assert trained.count(tokenizer.end_turn_id) == 2


def test_inference_template_ends_ready_for_the_assistant(tokenizer):
    ids = format_chat_for_inference(
        [{"role": "user", "content": "Hi"}], tokenizer=tokenizer
    )
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.assistant_id  # ready to generate the reply


def test_string_template_is_readable():
    s = format_chat_for_inference(
        [
            {"role": "system", "content": "You are Aila Nano."},
            {"role": "user", "content": "Hello!"},
        ]
    )
    assert s == "<s><|system|>You are Aila Nano.<|end|><|user|>Hello!<|end|><|assistant|>"


# -- the shipped datasets -----------------------------------------------------


def test_every_shipped_conversation_is_valid():
    files = list(CONV_DIR.glob("*.jsonl"))
    assert files, "expected conversational datasets to exist"
    for path in files:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            assert validate_conversation(record) is None, f"{path.name}:{line_no}"


def test_shipped_conversations_encode_without_dropping(tokenizer):
    for path in CONV_DIR.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            conv = ChatConversation.from_dict(json.loads(line))
            result = encode_conversation(conv, tokenizer, max_len=1024)
            assert result is not None


# -- the dataset --------------------------------------------------------------


def test_chat_dataset_loads_and_skips_invalid(tokenizer, tmp_path):
    from finetuning.chat_dataset import ChatDataset

    path = tmp_path / "c.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hey!"}]}),
                "{ not json",  # skipped
                json.dumps({"messages": [{"role": "user", "content": "x"}]}),  # invalid: no assistant
                json.dumps({"messages": [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "See you!"}]}),
            ]
        ),
        encoding="utf-8",
    )
    ds = ChatDataset(str(path), tokenizer, max_seq_len=1024)
    assert len(ds) == 2  # only the two valid conversations
    item = ds[0]
    # Shifted: input and labels are the same length, and something is trained.
    assert item["input_ids"].shape == item["labels"].shape
    assert (item["labels"] != -100).sum().item() > 0


def test_chat_dataset_reads_the_shipped_conversations(tokenizer):
    from finetuning.chat_dataset import ChatDataset

    files = [str(p) for p in CONV_DIR.glob("*.jsonl")]
    ds = ChatDataset(files, tokenizer, max_seq_len=1024)
    assert len(ds) >= 50
