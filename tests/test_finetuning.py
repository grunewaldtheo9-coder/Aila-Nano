from pathlib import Path

from finetuning.dataset import InstructionDataset, collate_instruction_batch
from finetuning.finetune import FinetuneConfig, run_finetune
from finetuning.format import IGNORE_INDEX, InstructionExample, encode_example
from model.transformer import AilaNanoGPT
from tests.conftest import AILA_KNOWLEDGE, SAMPLE_FINETUNE


def test_encode_example_masks_prompt_not_response(tokenizer):
    example = InstructionExample(instruction="Say hi.", output="Hi there!")
    ids, labels = encode_example(example, tokenizer)

    assert len(ids) == len(labels)
    # Everything up to and including the <|assistant|> token must be masked.
    assistant_pos = ids.index(tokenizer.assistant_id)
    assert all(label == IGNORE_INDEX for label in labels[: assistant_pos + 1])
    # At least one response token must be trainable (not masked).
    assert any(label != IGNORE_INDEX for label in labels[assistant_pos + 1 :])
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id


def test_encode_example_includes_system_and_input(tokenizer):
    example = InstructionExample(
        instruction="Translate.", input="Hello", output="Bonjour", system="Be terse."
    )
    ids, _ = encode_example(example, tokenizer)
    assert tokenizer.system_id in ids
    assert tokenizer.user_id in ids
    assert tokenizer.assistant_id in ids


def test_instruction_dataset_loads_aila_knowledge(tokenizer):
    ds = InstructionDataset(str(AILA_KNOWLEDGE), tokenizer, max_seq_len=256)
    assert len(ds) > 0
    item = ds[0]
    assert "input_ids" in item and "labels" in item
    assert item["input_ids"].shape == item["labels"].shape


def test_collate_pads_batch_to_max_length(tokenizer):
    ds = InstructionDataset(str(SAMPLE_FINETUNE), tokenizer, max_seq_len=256)
    batch = [ds[0], ds[1]]
    collated = collate_instruction_batch(batch, pad_id=tokenizer.pad_id)
    max_len = max(item["input_ids"].shape[0] for item in batch)
    assert collated["input_ids"].shape == (2, max_len)
    assert collated["labels"].shape == (2, max_len)


def test_encode_example_returns_none_when_response_cannot_fit(tokenizer):
    # Regression test: an example whose response alone exceeds max_len must
    # be rejected outright, never silently truncated down to zero
    # trainable labels (which used to produce a nan loss — see
    # finetuning/dataset.py).
    long_output = " ".join(["word"] * 100)
    example = InstructionExample(instruction="Say something long.", output=long_output)
    result = encode_example(example, tokenizer, max_len=8)
    assert result is None


def test_encode_example_truncates_from_the_front_preserving_response(tokenizer):
    example = InstructionExample(
        instruction="This is a fairly long instruction that will not fully fit.",
        system="This is an equally long system prompt padding out the prompt.",
        output="Short reply.",
    )
    full_ids, full_labels = encode_example(example, tokenizer, max_len=None)
    assistant_pos = full_ids.index(tokenizer.assistant_id)
    response_len = len(full_ids) - assistant_pos

    # Pick a budget with just enough slack over the response to prove
    # truncation happened, but nowhere near enough to keep the whole prompt.
    budget = response_len + 3
    assert budget < len(full_ids), "test setup: example must actually need truncating"

    result = encode_example(example, tokenizer, max_len=budget)
    assert result is not None
    ids, labels = result
    assert len(ids) == budget
    # The response must survive truncation — at least one trainable label.
    assert any(label != IGNORE_INDEX for label in labels)
    # And the tail must be exactly the original response tokens, unaltered.
    assert ids[-response_len:] == full_ids[-response_len:]
    assert labels[-response_len:] == full_labels[-response_len:]


def test_instruction_dataset_skips_unfittable_examples(tmp_path, tokenizer):
    import json

    path = tmp_path / "data.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"instruction": "hi", "output": "short reply"}) + "\n")
        f.write(json.dumps({"instruction": "hi", "output": " ".join(["word"] * 200)}) + "\n")

    ds = InstructionDataset(str(path), tokenizer, max_seq_len=16)
    # The oversized example must be dropped, not crash or corrupt the batch.
    assert len(ds) == 1
    item = ds[0]
    assert (item["labels"] != IGNORE_INDEX).any()


def test_run_finetune_end_to_end(tmp_path, tiny_model: AilaNanoGPT, tokenizer):
    cfg = FinetuneConfig(
        epochs=2,
        batch_size=2,
        max_lr=1e-3,
        min_lr=1e-4,
        out_dir=str(tmp_path / "ft"),
        device="cpu",
        val_fraction=0.2,
        log_interval=100,
    )
    run_finetune(tiny_model, tokenizer, [str(SAMPLE_FINETUNE)], cfg)
    assert (Path(cfg.out_dir) / "epoch_001.pt").exists()


def test_run_finetune_prunes_old_epoch_checkpoints(tmp_path, tiny_model: AilaNanoGPT, tokenizer):
    # Regression test: a many-epoch fine-tune (perfectly reasonable on a
    # small dataset, since fine-tuning is cheap) used to write one
    # never-deleted checkpoint per epoch and could fill the disk.
    cfg = FinetuneConfig(
        epochs=10,
        batch_size=2,
        max_lr=1e-3,
        min_lr=1e-4,
        out_dir=str(tmp_path / "ft"),
        device="cpu",
        val_fraction=0.2,
        log_interval=1000,
        keep_last_n_checkpoints=3,
    )
    run_finetune(tiny_model, tokenizer, [str(SAMPLE_FINETUNE)], cfg)

    epoch_checkpoints = sorted(Path(cfg.out_dir).glob("epoch_*.pt"))
    assert len(epoch_checkpoints) == 3
    # The retained checkpoints must be the *most recent* ones.
    assert [p.name for p in epoch_checkpoints] == ["epoch_007.pt", "epoch_008.pt", "epoch_009.pt"]
    assert (Path(cfg.out_dir) / "best.pt").exists()
