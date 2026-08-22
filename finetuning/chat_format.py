"""Multi-turn chat conversation format.

The single-turn instruction format (finetuning/format.py) teaches one
question -> one answer. Genuine conversation needs *multi-turn* dialogue:
context, follow-up questions, references to earlier messages. This module
encodes the spec's `messages` schema —

    {"messages": [
        {"role": "system",    "content": "You are Aila Nano..."},
        {"role": "user",      "content": "Hi!"},
        {"role": "assistant", "content": "Hey! How are you?"},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}

— using the **same** special tokens as the single-turn format
(<|system|> <|user|> <|assistant|> <|end|>, BOS/EOS), so a model can be
trained on both and the chat template at inference matches training
exactly. Every assistant turn contributes to the loss; system and user
turns are masked with IGNORE_INDEX. The final assistant turn is closed
with EOS so the model learns to stop after replying (spec §19).

The same layout renders the canonical Aila Nano chat template:

    <s><|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|>...</s>
"""

from __future__ import annotations

from dataclasses import dataclass

from finetuning.format import IGNORE_INDEX
from tokenizer.tokenizer import AilaTokenizer

VALID_ROLES = ("system", "user", "assistant")


@dataclass
class ChatConversation:
    messages: list[dict]  # each: {"role": ..., "content": ...}

    @classmethod
    def from_dict(cls, d: dict) -> "ChatConversation":
        msgs = d.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise ValueError("conversation must have a non-empty 'messages' list")
        cleaned = []
        for m in msgs:
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if role not in VALID_ROLES:
                raise ValueError(f"invalid role: {role!r}")
            if not content:
                raise ValueError("empty message content")
            cleaned.append({"role": role, "content": content})
        return cls(messages=cleaned)


def validate_conversation(d: dict) -> str | None:
    """Return None if `d` is a well-formed conversation, else a short
    reason string. A dataset loader uses this to reject bad records rather
    than crash training halfway through (spec §29).

    Rules: a non-empty `messages` list, valid alternating roles (an
    optional leading system turn, then user/assistant strictly
    alternating), non-empty content, at least one assistant turn, and no
    two same-role turns in a row.
    """
    msgs = d.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return "missing or empty 'messages'"
    i = 0
    if msgs[0].get("role") == "system":
        if not (msgs[0].get("content") or "").strip():
            return "empty system content"
        i = 1
    if i >= len(msgs):
        return "no user/assistant turns"
    expected = "user"  # a conversation starts with the user after any system
    saw_assistant = False
    for m in msgs[i:]:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if role == "system":
            return "system turn must be first"
        if role not in ("user", "assistant"):
            return f"invalid role: {role!r}"
        if not content:
            return "empty message content"
        if role != expected:
            return f"roles must alternate user/assistant (got {role} where {expected} expected)"
        if role == "assistant":
            saw_assistant = True
        expected = "assistant" if expected == "user" else "user"
    if not saw_assistant:
        return "no assistant turn to train on"
    return None


def encode_conversation(
    conv: ChatConversation, tokenizer: AilaTokenizer, max_len: int | None = None
) -> tuple[list[int], list[int]] | None:
    """Encode a whole conversation to (input_ids, labels). Every assistant
    turn is trained on (its content + closing tokens); system/user turns are
    masked with IGNORE_INDEX. Returns None if no assistant span survives the
    length limit (front-truncation, like the single-turn encoder).
    """
    ids: list[int] = [tokenizer.bos_id]
    labels: list[int] = [IGNORE_INDEX]
    n_msgs = len(conv.messages)

    for idx, msg in enumerate(conv.messages):
        role, content = msg["role"], msg["content"]
        content_ids = tokenizer.encode(content)
        is_last = idx == n_msgs - 1
        if role == "system":
            ids += [tokenizer.system_id] + content_ids + [tokenizer.end_turn_id]
            labels += [IGNORE_INDEX] * (len(content_ids) + 2)
        elif role == "user":
            ids += [tokenizer.user_id] + content_ids + [tokenizer.end_turn_id]
            labels += [IGNORE_INDEX] * (len(content_ids) + 2)
        else:  # assistant — trained on
            closing = [tokenizer.end_turn_id]
            if is_last:
                closing.append(tokenizer.eos_id)  # learn to stop after the last reply
            ids += [tokenizer.assistant_id] + content_ids + closing
            # Not the introducing <|assistant|> token itself — it's a cue.
            labels += [IGNORE_INDEX] + content_ids + closing

    if not any(l != IGNORE_INDEX for l in labels):
        return None

    if max_len is not None and len(ids) > max_len:
        # Front-truncate, but only if an assistant label survives.
        keep_from = len(ids) - max_len + 1
        new_ids = [ids[0]] + ids[keep_from:]
        new_labels = [labels[0]] + labels[keep_from:]
        if not any(l != IGNORE_INDEX for l in new_labels):
            return None
        ids, labels = new_ids, new_labels

    return ids, labels


def format_chat_for_inference(
    messages: list[dict], tokenizer: AilaTokenizer | None = None
) -> list[int] | str:
    """Build the prompt (up to and including a trailing <|assistant|>) for
    generation, from a running list of {"role","content"} messages. The
    string form (tokenizer=None) is the human-readable chat template."""
    if tokenizer is None:
        parts = ["<s>"]
        for m in messages:
            role, content = m["role"], m["content"]
            tag = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}[role]
            parts.append(f"{tag}{content}<|end|>")
        parts.append("<|assistant|>")
        return "".join(parts)

    ids = [tokenizer.bos_id]
    tag_id = {
        "system": tokenizer.system_id,
        "user": tokenizer.user_id,
        "assistant": tokenizer.assistant_id,
    }
    for m in messages:
        ids += [tag_id[m["role"]]] + tokenizer.encode(m["content"]) + [tokenizer.end_turn_id]
    ids += [tokenizer.assistant_id]
    return ids
