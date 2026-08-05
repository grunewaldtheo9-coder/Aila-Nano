"""Instruction-tuning example format.

Each training example is a JSON object (one per line, JSONL):

    {"instruction": "...", "input": "...", "output": "...", "system": "..."}

`input` and `system` are optional. `instruction` + `input` (if present,
joined by a newline) become the user turn; `output` becomes the
assistant turn the model is trained to produce.

Tokens are laid out as:

    <s> [<|system|> system <|end|>] <|user|> instruction[\\n input] <|end|> <|assistant|> output <|end|> </s>

Only the assistant's output (plus its closing tokens) contributes to the
loss — the prompt is masked with label id -100 — which is standard
instruction-tuning practice and keeps the model from wasting capacity
learning to *predict the prompt* instead of *responding to it*.
"""

from __future__ import annotations

from dataclasses import dataclass

from tokenizer.tokenizer import AilaTokenizer

IGNORE_INDEX = -100


@dataclass
class InstructionExample:
    instruction: str
    output: str
    input: str = ""
    system: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> InstructionExample:
        return cls(
            instruction=d["instruction"],
            output=d["output"],
            input=d.get("input", "") or "",
            system=d.get("system", "") or "",
        )

    @property
    def user_text(self) -> str:
        if self.input:
            return f"{self.instruction}\n{self.input}"
        return self.instruction


def encode_example(
    example: InstructionExample, tokenizer: AilaTokenizer, max_len: int | None = None
) -> tuple[list[int], list[int]] | None:
    """Returns (input_ids, labels) — parallel lists, labels has IGNORE_INDEX
    everywhere except the assistant's output span — or None if the example
    cannot be made to fit within `max_len` without truncating away its
    entire trainable (assistant) span.

    If the full example exceeds `max_len`, tokens are dropped from the
    *front* (the earliest system/user context) rather than the end. Naively
    truncating from the end would risk chopping off the assistant response
    itself, leaving an example with zero trainable labels — which silently
    produces a `nan` loss for that example (cross-entropy over an empty
    target set) and can poison an entire training step. Truncating from the
    front instead means we lose some prompt context on an overlong example
    (still not ideal) rather than corrupting training.
    """
    ids: list[int] = [tokenizer.bos_id]
    labels: list[int] = [IGNORE_INDEX]

    if example.system:
        sys_ids = tokenizer.encode(example.system)
        ids += [tokenizer.system_id] + sys_ids + [tokenizer.end_turn_id]
        labels += [IGNORE_INDEX] * (len(sys_ids) + 2)

    user_ids = tokenizer.encode(example.user_text)
    ids += [tokenizer.user_id] + user_ids + [tokenizer.end_turn_id]
    labels += [IGNORE_INDEX] * (len(user_ids) + 2)

    response_start = len(ids)  # index of the <|assistant|> token
    out_ids = tokenizer.encode(example.output)
    ids += [tokenizer.assistant_id] + out_ids + [tokenizer.end_turn_id, tokenizer.eos_id]
    # Train on the output tokens themselves and the two closing tokens, but
    # not on the <|assistant|> token that *introduces* the turn (it's a
    # cue the model reads, not something meaningful to predict).
    labels += [IGNORE_INDEX] + out_ids + [tokenizer.end_turn_id, tokenizer.eos_id]

    if max_len is not None and len(ids) > max_len:
        response_len = len(ids) - response_start
        if response_len > max_len:
            # Even the assistant span alone doesn't fit — no amount of
            # front-truncation can save this example.
            return None
        # Keep the BOS token plus as much of the tail (which always
        # includes the full response) as fits.
        keep_from = len(ids) - max_len + 1
        ids = [ids[0]] + ids[keep_from:]
        labels = [labels[0]] + labels[keep_from:]

    return ids, labels


def format_prompt_for_inference(
    instruction: str,
    input_text: str = "",
    system: str = "",
    tokenizer: AilaTokenizer | None = None,
) -> list[int] | str:
    """Build the prompt token ids (or raw string, if tokenizer is None) up
    to and including <|assistant|>, ready for `model.generate`.
    """
    example = InstructionExample(instruction=instruction, input=input_text, output="", system=system)
    if tokenizer is None:
        parts = []
        if system:
            parts.append(f"<|system|>{system}<|end|>")
        parts.append(f"<|user|>{example.user_text}<|end|><|assistant|>")
        return "<s>" + "".join(parts)

    ids = [tokenizer.bos_id]
    if system:
        ids += [tokenizer.system_id] + tokenizer.encode(system) + [tokenizer.end_turn_id]
    ids += [tokenizer.user_id] + tokenizer.encode(example.user_text) + [tokenizer.end_turn_id]
    ids += [tokenizer.assistant_id]
    return ids
