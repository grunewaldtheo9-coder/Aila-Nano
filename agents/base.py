"""Base Agent: all specialized assistants share the same underlying
Aila Nano model and tokenizer, and differ only in system prompt / default
generation parameters — exactly as the project spec requires ("Each agent
should share the same LLM but use different system behaviors.").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from finetuning.format import format_prompt_for_inference
from memory.manager import MemoryContext, MemoryManager
from model.generate import generate, generate_stream
from model.transformer import AilaNanoGPT
from tokenizer.tokenizer import AilaTokenizer

AILA_KNOWLEDGE_PRIMER = (
    "You are Aila Nano, a small language model created by Aila Company Solutions, "
    "founded by Theo Grunewald Hames and Guilherme Grunewald Benkendorf. "
)


@dataclass
class GenerationSettings:
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int | None = 40
    top_p: float | None = 0.95
    repetition_penalty: float = 1.15


class Agent:
    """Base class for a specialized assistant persona."""

    name: str = "assistant"
    system_prompt: str = AILA_KNOWLEDGE_PRIMER + "You are a helpful, honest assistant."
    default_settings: GenerationSettings = GenerationSettings()

    def __init__(
        self,
        model: AilaNanoGPT,
        tokenizer: AilaTokenizer,
        memory: MemoryManager | None = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.memory = memory
        self.device = device

    # -- prompt construction -----------------------------------------------

    def _build_system_prompt(self, memory_ctx: MemoryContext | None) -> str:
        prompt = self.system_prompt
        if memory_ctx and memory_ctx.relevant_facts:
            facts = "\n".join(f"- {f['content']}" for f in memory_ctx.relevant_facts)
            prompt = f"{prompt}\n\nRelevant background you may use if helpful:\n{facts}"
        return prompt

    def _build_prompt_ids(
        self, conversation_id: str, user_message: str, memory_ctx: MemoryContext | None
    ) -> list[int]:
        tok = self.tokenizer
        ids = [tok.bos_id]
        system = self._build_system_prompt(memory_ctx)
        ids += [tok.system_id] + tok.encode(system) + [tok.end_turn_id]

        if memory_ctx:
            for turn in memory_ctx.history:
                role_id = tok.user_id if turn["role"] == "user" else tok.assistant_id
                ids += [role_id] + tok.encode(turn["content"]) + [tok.end_turn_id]

        ids += [tok.user_id] + tok.encode(user_message) + [tok.end_turn_id]
        ids += [tok.assistant_id]

        # Truncate from the *left* (drop oldest history) if the prompt would
        # overflow the model's context window. Reserve at least a small
        # generation budget so the model always has room to reply, even in
        # constrained configs where max_new_tokens is close to max_seq_len.
        reserve = min(self.default_settings.max_new_tokens, self.model.cfg.max_seq_len // 2)
        max_prompt_len = max(1, self.model.cfg.max_seq_len - reserve)
        if len(ids) > max_prompt_len:
            ids = ids[-max_prompt_len:]
        return ids

    # -- inference -----------------------------------------------------

    @torch.no_grad()
    def respond(
        self,
        conversation_id: str,
        user_message: str,
        settings: GenerationSettings | None = None,
        remember_turn: bool = True,
    ) -> str:
        settings = settings or self.default_settings
        memory_ctx = (
            self.memory.build_context(conversation_id, query=user_message)
            if self.memory
            else None
        )

        prompt_ids = self._build_prompt_ids(conversation_id, user_message, memory_ctx)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        out = generate(
            self.model,
            input_tensor,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_k=settings.top_k,
            top_p=settings.top_p,
            repetition_penalty=settings.repetition_penalty,
            eos_id=self.tokenizer.end_turn_id,
        )
        new_ids = out[0, len(prompt_ids) :].tolist()
        reply = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        if self.memory and remember_turn:
            self.memory.add_turn(conversation_id, "user", user_message, agent_type=self.name)
            self.memory.add_turn(conversation_id, "assistant", reply, agent_type=self.name)

        return reply

    def respond_stream(
        self,
        conversation_id: str,
        user_message: str,
        settings: GenerationSettings | None = None,
        remember_turn: bool = True,
    ):
        """Generator form of `respond`: yields decoded text deltas as they
        are produced, and stores the full turn in memory once generation
        finishes. Used by the streaming (SSE) chat endpoint.
        """
        settings = settings or self.default_settings
        memory_ctx = (
            self.memory.build_context(conversation_id, query=user_message)
            if self.memory
            else None
        )

        prompt_ids = self._build_prompt_ids(conversation_id, user_message, memory_ctx)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        produced_ids: list[int] = []
        decoded_so_far = ""
        for token_id in generate_stream(
            self.model,
            input_tensor,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_k=settings.top_k,
            top_p=settings.top_p,
            repetition_penalty=settings.repetition_penalty,
            eos_id=self.tokenizer.end_turn_id,
        ):
            if token_id == self.tokenizer.end_turn_id:
                break
            produced_ids.append(token_id)
            decoded_full = self.tokenizer.decode(produced_ids, skip_special_tokens=True)
            delta = decoded_full[len(decoded_so_far) :]
            decoded_so_far = decoded_full
            if delta:
                yield delta

        if self.memory and remember_turn:
            self.memory.add_turn(conversation_id, "user", user_message, agent_type=self.name)
            self.memory.add_turn(conversation_id, "assistant", decoded_so_far, agent_type=self.name)

    def prompt_preview(self, conversation_id: str, user_message: str) -> str:
        """Debug helper: render the prompt as a string instead of ids."""
        memory_ctx = (
            self.memory.build_context(conversation_id, query=user_message)
            if self.memory
            else None
        )
        system = self._build_system_prompt(memory_ctx)
        return format_prompt_for_inference(user_message, system=system)
