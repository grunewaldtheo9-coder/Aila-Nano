"""Base Agent: all specialized assistants share the same underlying
Aila Nano model and tokenizer, and differ only in system prompt / default
generation parameters — exactly as the project spec requires ("Each agent
should share the same LLM but use different system behaviors.").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from finetuning.format import format_prompt_for_inference
from memory.commands import guess_category, parse_memory_command
from memory.lexical import lexical_overlap_score
from memory.manager import MemoryContext, MemoryManager
from model.generate import DEFAULT_NO_REPEAT_NGRAM_SIZE, generate, generate_stream
from model.transformer import AilaNanoGPT
from tokenizer.tokenizer import AilaTokenizer
from vectordb.semantic_index import SemanticIndex

AILA_KNOWLEDGE_PRIMER = (
    "You are Aila Nano, a small language model created by Aila Company Solutions, "
    "founded by Theo Grunewald Hames and Guilherme Grunewald Benkendorf. "
)

# How many knowledge-base chunks (from files indexed via `AilaEngine.learn_file`
# / the terminal's `/learn` command) to surface as context per turn.
KNOWLEDGE_TOP_K = 2

# How much word-overlap (memory/lexical.py) an explicit "forget that X"
# command needs against a stored memory before it's deleted. Higher than
# the general injection threshold (memory/semantic_memory.py's
# DEFAULT_RELEVANCE_THRESHOLD) on purpose — deleting is destructive, so a
# vague "forget that" should say "nothing matched" rather than guess and
# remove the wrong memory.
FORGET_MATCH_THRESHOLD = 0.34

# Cap on how many memories "what do you remember about me?" lists at once,
# so a long-lived install with hundreds of stored facts still gets a
# readable reply instead of a wall of text.
LIST_MEMORIES_LIMIT = 20


@dataclass
class GenerationSettings:
    max_new_tokens: int = 200
    # A ~10.9M-param model trained on a modest instruction set has a much
    # peakier, less reliable next-token distribution than a large model —
    # the same temperature/top_k/top_p that reads as "reasonably creative"
    # on a big model lets this one wander into incoherent territory well
    # before max_new_tokens is reached (confirmed via direct comparison:
    # the same checkpoint stays on-topic far more often at these tighter
    # settings than at the previous temperature=0.8/top_k=40/top_p=0.95).
    temperature: float = 0.2
    top_k: int | None = 4
    top_p: float | None = None
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE


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
        knowledge: SemanticIndex | None = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.memory = memory
        self.knowledge = knowledge
        self.device = device

    # -- prompt construction -----------------------------------------------

    def _build_system_prompt(self, query: str, memory_ctx: MemoryContext | None) -> str:
        prompt = self.system_prompt

        # Only ever appended when memory_ctx.relevant_facts is non-empty —
        # memory.semantic_memory.get_relevant_memories returns [] rather
        # than "the closest thing anyway" when nothing clears the
        # relevance threshold, so "no [MEMORY] block" is a real, reachable
        # state precisely when no relevant memory exists (goal: never
        # invent one). Bracketed [MEMORY]/[/MEMORY] tags keep the injected
        # facts visually distinct from the rest of the system prompt.
        if memory_ctx and memory_ctx.relevant_facts:
            facts = "\n".join(f"- {f['content']}" for f in memory_ctx.relevant_facts)
            prompt = f"{prompt}\n\n[MEMORY]\n{facts}\n[/MEMORY]"

        if self.knowledge and len(self.knowledge) > 0:
            hits = self.knowledge.search(query, k=KNOWLEDGE_TOP_K)
            if hits:
                snippets = "\n".join(f"- {h['text']}" for h in hits)
                prompt = f"{prompt}\n\nRelevant notes from your knowledge base:\n{snippets}"

        return prompt

    def _build_prompt_ids(
        self, user_message: str, memory_ctx: MemoryContext | None
    ) -> list[int]:
        tok = self.tokenizer
        ids = [tok.bos_id]
        system = self._build_system_prompt(user_message, memory_ctx)
        ids += [tok.system_id] + tok.encode(system) + [tok.end_turn_id]

        # Deliberately NOT injecting memory_ctx.history as extra
        # <|user|>/<|assistant|> turns here, even though the tokens exist to
        # do it. Every fine-tuning example is a single system+user+assistant
        # turn — the model has never seen more than one user/assistant pair
        # in a prompt. Feeding real prior turns in makes the input
        # increasingly out-of-distribution as a conversation grows, and in
        # practice knocks the model off "answer as Aila Nano" and back onto
        # its pretraining prior (freeform TinyStories-style narrative) after
        # just a couple of turns. Conversation history is still recorded via
        # `_remember_turn` below (so `/history` and long-term memory keep
        # working) — it's just not replayed into the generation prompt
        # until the model is actually fine-tuned on multi-turn examples.
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

    def _prepare_turn(
        self, conversation_id: str, user_message: str
    ) -> tuple[torch.Tensor, list[int]]:
        """Shared setup for `respond`/`respond_stream`: build memory
        context, then the prompt ids, then the input tensor. Returns
        (input_tensor, prompt_ids).
        """
        memory_ctx = (
            self.memory.build_context(conversation_id, query=user_message)
            if self.memory
            else None
        )
        prompt_ids = self._build_prompt_ids(user_message, memory_ctx)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        return input_tensor, prompt_ids

    def _remember_turn(self, conversation_id: str, user_message: str, reply: str) -> None:
        if self.memory:
            self.memory.add_turn(conversation_id, "user", user_message, agent_type=self.name)
            self.memory.add_turn(conversation_id, "assistant", reply, agent_type=self.name)

    # -- explicit memory commands -------------------------------------------

    def _handle_memory_command(self, user_message: str) -> str | None:
        """Deterministically handle "remember that X" / "forget that X" /
        "what do you remember about me?" without ever calling the model —
        the one guaranteed-zero-hallucination path in the system. Returns
        the reply text if `user_message` was a recognized command, else
        None (caller falls through to normal generation).
        """
        if not self.memory:
            return None
        command = parse_memory_command(user_message)
        if command is None:
            return None

        if command.kind == "remember":
            category = guess_category(command.content)
            self.memory.add_memory(command.content, category=category, importance=0.8)
            return f"Got it — I'll remember that {command.content}."

        if command.kind == "forget":
            facts = self.memory.all_memories()
            scored = sorted(
                facts, key=lambda f: lexical_overlap_score(command.content, f["content"]), reverse=True
            )
            best = scored[0] if scored else None
            if best and lexical_overlap_score(command.content, best["content"]) >= FORGET_MATCH_THRESHOLD:
                self.memory.delete_memory(best["id"])
                return f"Done — I've forgotten that {best['content']}"
            return "I don't have anything remembered that matches that."

        if command.kind == "list":
            facts = self.memory.all_memories()
            if not facts:
                return "I don't have anything remembered yet."
            facts = sorted(facts, key=lambda f: f["created_at"], reverse=True)[:LIST_MEMORIES_LIMIT]
            lines = "\n".join(f"- {f['content']}" for f in facts)
            return f"Here's what I remember:\n{lines}"

        return None

    # -- inference -----------------------------------------------------

    @torch.no_grad()
    def respond(
        self,
        conversation_id: str,
        user_message: str,
        settings: GenerationSettings | None = None,
        remember_turn: bool = True,
    ) -> str:
        command_reply = self._handle_memory_command(user_message)
        if command_reply is not None:
            if remember_turn:
                self._remember_turn(conversation_id, user_message, command_reply)
            return command_reply

        settings = settings or self.default_settings
        input_tensor, prompt_ids = self._prepare_turn(conversation_id, user_message)

        out = generate(
            self.model,
            input_tensor,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_k=settings.top_k,
            top_p=settings.top_p,
            repetition_penalty=settings.repetition_penalty,
            no_repeat_ngram_size=settings.no_repeat_ngram_size,
            eos_id=self.tokenizer.end_turn_id,
            suppress_token_ids=self.tokenizer.byte_fallback_ids,
        )
        new_ids = out[0, len(prompt_ids) :].tolist()
        reply = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        if remember_turn:
            self._remember_turn(conversation_id, user_message, reply)

        return reply

    @torch.no_grad()
    def respond_stream(
        self,
        conversation_id: str,
        user_message: str,
        settings: GenerationSettings | None = None,
        remember_turn: bool = True,
    ):
        """Generator form of `respond`: yields decoded text deltas as they
        are produced, and stores the full turn in memory once generation
        finishes. This is what `chat.py` uses to print Aila's reply as it
        types, rather than waiting for the whole response.
        """
        command_reply = self._handle_memory_command(user_message)
        if command_reply is not None:
            yield command_reply
            if remember_turn:
                self._remember_turn(conversation_id, user_message, command_reply)
            return

        settings = settings or self.default_settings
        input_tensor, _ = self._prepare_turn(conversation_id, user_message)

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
            no_repeat_ngram_size=settings.no_repeat_ngram_size,
            eos_id=self.tokenizer.end_turn_id,
            suppress_token_ids=self.tokenizer.byte_fallback_ids,
        ):
            if token_id == self.tokenizer.end_turn_id:
                break
            produced_ids.append(token_id)
            decoded_full = self.tokenizer.decode(produced_ids, skip_special_tokens=True)
            delta = decoded_full[len(decoded_so_far) :]
            decoded_so_far = decoded_full
            if delta:
                yield delta

        if remember_turn:
            self._remember_turn(conversation_id, user_message, decoded_so_far)

    def prompt_preview(self, conversation_id: str, user_message: str) -> str:
        """Debug helper: render the prompt as a string instead of ids."""
        memory_ctx = (
            self.memory.build_context(conversation_id, query=user_message)
            if self.memory
            else None
        )
        system = self._build_system_prompt(user_message, memory_ctx)
        return format_prompt_for_inference(user_message, system=system)
