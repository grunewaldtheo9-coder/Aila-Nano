"""Base Agent: all specialized assistants share the same underlying
Aila Nano model and tokenizer, and differ only in system prompt / default
generation parameters — exactly as the project spec requires ("Each agent
should share the same LLM but use different system behaviors.").
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

from finetuning.format import format_prompt_for_inference
from memory.commands import guess_category, parse_memory_command
from memory.lexical import lexical_overlap_score, tokenize
from memory.manager import MemoryContext, MemoryManager
from model.generate import DEFAULT_NO_REPEAT_NGRAM_SIZE, generate, generate_stream
from model.transformer import AilaNanoGPT
from tokenizer.tokenizer import AilaTokenizer
from tools.smalltalk import match_smalltalk
from webresearch.pipeline import detect_language
from vectordb.semantic_index import SemanticIndex

logger = logging.getLogger(__name__)

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

# Attribute words that are too generic to identify *which* memory to forget
# on their own: "forget my favorite movie" must not match "my favorite game"
# on the shared word "favorite". Treated as generic when checking that a
# forget request shares a distinctive term with a stored memory.
_FORGET_GENERIC_TERMS = frozenset(
    {"favorite", "favourite", "favorito", "favorita", "favoritos", "favoritas"}
)

# Cap on how many memories "what do you remember about me?" lists at once,
# so a long-lived install with hundreds of stored facts still gets a
# readable reply instead of a wall of text.
LIST_MEMORIES_LIMIT = 20

# What Aila says instead of generating freeform text, when freeform
# generation is switched off (which it is by default — see
# `Agent.fallback_reply`).
NO_FREEFORM_REPLY_EN = (
    "I'm not sure how to answer that one. I'm a small model in beta, so I'm "
    "best at:\n"
    "  - questions I can look up (\"Who founded Apple?\")\n"
    "  - exact maths (\"What is 45 / 9?\")\n"
    "  - remembering things you tell me (\"remember that my name is Theo\")\n"
    "  - questions about myself (\"Who created you?\")\n"
    "Try one of those, or type /help."
)
NO_FREEFORM_REPLY_PT = (
    "Não sei responder isso. Sou um modelo pequeno em beta, então me saio "
    "melhor com:\n"
    "  - perguntas que eu posso pesquisar (\"Quem criou a Petrobras?\")\n"
    "  - contas exatas (\"Quanto é 45 / 9?\")\n"
    "  - lembrar coisas que você me conta (\"remember that my name is Theo\")\n"
    "  - perguntas sobre mim (\"Quem criou você?\")\n"
    "Tente uma dessas, ou digite /help."
)


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
        router=None,  # tools.router.ToolRouter | None (kept untyped to avoid an import cycle)
        allow_freeform: bool = True,
        translator=None,  # translation.Translator | None
        emit_status=None,  # Callable[[str], None] | None
        conversation_manager=None,  # conversation.ConversationManager | None
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.memory = memory
        self.knowledge = knowledge
        self.device = device
        self.router = router
        # The canonical context assembler (summary + relevant memories +
        # recent turns). When present it is the source of the generation
        # context; without it the agent falls back to memory.build_context.
        self.conversation_manager = conversation_manager
        # Optional en<->pt translation, used only as a *fallback* — see
        # `_resolve_route`. None (or an unavailable translator) simply
        # means Portuguese questions are handled natively only.
        self.translator = translator
        # Short progress lines ("Translating...") for a caller to show, so
        # the multi-second translate+research fallback isn't a silent
        # pause. Defaults to a no-op.
        self._emit_status = emit_status or (lambda msg: None)
        # When False, a message the router could not handle gets an honest
        # "here is what I can actually do" reply instead of freeform
        # generation.
        #
        # Measured, not assumed: sampling this checkpoint on twelve
        # prompts it was *fine-tuned on* produced usable text three times.
        # "Write a Python function that reverses a string." returned
        # "Have a great day."; "What is a variable in programming?"
        # returned "A capital do Ficoect ajudar?". At ~20M parameters
        # trained on ~11.5M tokens, freeform generation is noise, and
        # noise is worse than an honest answer.
        #
        # Defaults to True here so the class stays a plain language-model
        # wrapper for tests and library use; `AilaEngine` sets it from
        # configuration (AILA_ALLOW_FREEFORM, default false).
        self.allow_freeform = allow_freeform

    def _no_freeform_reply(self, user_message: str) -> str:
        from webresearch.pipeline import detect_language

        return (
            NO_FREEFORM_REPLY_PT
            if detect_language(user_message) == "pt"
            else NO_FREEFORM_REPLY_EN
        )

    # -- prompt construction -----------------------------------------------

    def _build_system_prompt(
        self,
        query: str,
        memory_ctx: MemoryContext | None,
        web_snippets: list[str] | None = None,
    ) -> str:
        prompt = self.system_prompt

        # Only ever appended when memory_ctx.relevant_facts is non-empty —
        # memory.semantic_memory.get_relevant_memories returns [] rather
        # than "the closest thing anyway" when nothing clears the
        # relevance threshold, so "no [MEMORY] block" is a real, reachable
        # state precisely when no relevant memory exists (goal: never
        # invent one). Bracketed [MEMORY]/[/MEMORY] tags keep the injected
        # facts visually distinct from the rest of the system prompt.
        # A conversation summary of older turns (from ConversationManager)
        # keeps a long conversation coherent without replaying every raw
        # turn. Same bracketed-data framing as [MEMORY] — background context,
        # not instructions.
        if memory_ctx and getattr(memory_ctx, "summary", ""):
            prompt = f"{prompt}\n\n[SUMMARY]\n{memory_ctx.summary}\n[/SUMMARY]"

        if memory_ctx and memory_ctx.relevant_facts:
            facts = "\n".join(f"- {f['content']}" for f in memory_ctx.relevant_facts)
            prompt = f"{prompt}\n\n[MEMORY]\n{facts}\n[/MEMORY]"

        # Web research context (already sanitized by webresearch/quality.py
        # before it can reach here). Same bracketed-data framing as
        # [MEMORY]: retrieved text is background DATA, never instructions
        # — and it's only present when the router found something both
        # relevant and worth injecting.
        if web_snippets:
            joined = "\n".join(f"- {s}" for s in web_snippets)
            prompt = f"{prompt}\n\n[WEB]\n{joined}\n[/WEB]"

        if self.knowledge and len(self.knowledge) > 0:
            hits = self.knowledge.search(query, k=KNOWLEDGE_TOP_K)
            if hits:
                snippets = "\n".join(f"- {h['text']}" for h in hits)
                prompt = f"{prompt}\n\nRelevant notes from your knowledge base:\n{snippets}"

        return prompt

    def _build_prompt_ids(
        self,
        user_message: str,
        memory_ctx: MemoryContext | None,
        web_snippets: list[str] | None = None,
    ) -> list[int]:
        tok = self.tokenizer
        ids = [tok.bos_id]
        system = self._build_system_prompt(user_message, memory_ctx, web_snippets=web_snippets)
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
        self,
        conversation_id: str,
        user_message: str,
        web_snippets: list[str] | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        """Shared setup for `respond`/`respond_stream`: build memory
        context, then the prompt ids, then the input tensor. Returns
        (input_tensor, prompt_ids).
        """
        memory_ctx = self._assemble_context(conversation_id, user_message)
        prompt_ids = self._build_prompt_ids(user_message, memory_ctx, web_snippets=web_snippets)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        return input_tensor, prompt_ids

    def _assemble_context(self, conversation_id: str, user_message: str) -> MemoryContext | None:
        """Build the generation context. Prefers the ConversationManager
        (summary of older turns + relevant memories) so a long conversation
        stays coherent; falls back to the memory manager's context, then to
        nothing. This is the single canonical context source for generation."""
        if self.conversation_manager is not None:
            bundle = self.conversation_manager.assemble(
                conversation_id, user_message, max_facts=self.memory.max_facts if self.memory else 3
            )
            return MemoryContext(
                history=bundle.recent,
                relevant_facts=bundle.relevant_facts,
                summary=bundle.summary,
            )
        if self.memory:
            return self.memory.build_context(conversation_id, query=user_message)
        return None

    def _remember_turn(self, conversation_id: str, user_message: str, reply: str) -> None:
        if self.memory:
            self.memory.add_turn(conversation_id, "user", user_message, agent_type=self.name)
            self.memory.add_turn(conversation_id, "assistant", reply, agent_type=self.name)

    def _previous_user_message(self, conversation_id: str) -> str | None:
        """The user's most recent *substantive* prior turn, used by the
        tool router to resolve short follow-ups ("When?").

        Filler is skipped. A real transcript went:

            You: Who founded Bambu Lab?   -> answered
            You: Ok?                      -> "Got it."
            You: When?                    -> the Wikipedia article on "OK"

        because "Ok?" was the immediately preceding user turn, so the
        follow-up resolved against it. A follow-up means "more about what
        we were actually discussing", and an acknowledgement is not that.
        """
        if not self.memory:
            return None
        history = self.memory.conversation.get_history(conversation_id, max_turns=12)
        for turn in reversed(history):
            if turn["role"] != "user":
                continue
            content = (turn["content"] or "").strip()
            if not content or match_smalltalk(content) is not None:
                continue
            if not tokenize(content):
                continue  # nothing searchable to inherit
            return content
        return None

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

        # A Portuguese command ("lembre que ...", "esqueça ...") deserves a
        # Portuguese confirmation — replying in English to a message the user
        # wrote in Portuguese is jarring, and this path is the one place we
        # know for certain what the reply should say.
        pt = detect_language(user_message) == "pt"

        if command.kind == "remember":
            category = guess_category(command.content)
            # A "/remember" or "remember that ..." is the user explicitly
            # asking Aila to keep this — highest confidence and source. If it
            # names a versioned attribute, add_memory supersedes the old value.
            self.memory.add_memory(
                command.content,
                category=category,
                importance=0.8,
                source="explicit_user_request",
                confidence=0.95,
            )
            if pt:
                return f"Entendi — vou lembrar que {command.content}."
            return f"Got it — I'll remember that {command.content}."

        if command.kind == "forget":
            # First: "forget my favorite game" names an attribute with no
            # value — deactivate its current value so it stops being returned.
            from memory.attributes import extract_attribute_key
            from memory.lexical import GENERIC_QUESTION_TERMS, has_distinctive_overlap

            key = extract_attribute_key(command.content)
            if key is not None and hasattr(self.memory, "forget_attribute"):
                if self.memory.current_attribute(key) is not None:
                    self.memory.forget_attribute(key)
                    if pt:
                        return "Pronto — esqueci isso."
                    return "Done — I've forgotten that."

            # Lexical fallback for memories stored without an attribute key.
            # Require a shared *distinctive* term, treating attribute words
            # like "favorite" as generic — otherwise "forget my favorite
            # movie" shares only "favorite" with "my favorite game is Zelda"
            # and would delete the wrong memory.
            forget_generic = GENERIC_QUESTION_TERMS | _FORGET_GENERIC_TERMS
            facts = [
                f
                for f in self.memory.all_memories()
                if has_distinctive_overlap(command.content, f["content"], generic=forget_generic)
            ]
            scored = sorted(
                facts, key=lambda f: lexical_overlap_score(command.content, f["content"]), reverse=True
            )
            best = scored[0] if scored else None
            if best and lexical_overlap_score(command.content, best["content"]) >= FORGET_MATCH_THRESHOLD:
                self.memory.delete_memory(best["id"])
                if pt:
                    return f"Pronto — esqueci que {best['content']}"
                return f"Done — I've forgotten that {best['content']}"
            if pt:
                return "Não tenho nada guardado que combine com isso."
            return "I don't have anything remembered that matches that."

        if command.kind == "list":
            facts = self.memory.all_memories()
            if not facts:
                return "Ainda não guardei nada." if pt else "I don't have anything remembered yet."
            facts = sorted(facts, key=lambda f: f["created_at"], reverse=True)[:LIST_MEMORIES_LIMIT]
            lines = "\n".join(f"- {f['content']}" for f in facts)
            header = "Aqui está o que eu lembro:" if pt else "Here's what I remember:"
            return f"{header}\n{lines}"

        return None

    # -- routing (+ translation fallback) ----------------------------------

    # Router outcomes that ARE a `direct_reply` but really mean "I don't
    # have this" — a memory miss, or a search that found nothing. They are
    # answered natively (in Portuguese), but they must not block the
    # English translation retry: "Qual é o meu nome?" produces a
    # Portuguese memory-miss, yet the stored fact "my name is Theo" is
    # findable once the question is translated. A confident answer in
    # English is preferred over a soft miss in Portuguese.
    _SOFT_MISS_TOOLS = frozenset({"memory_miss", "web_no_answer"})

    @classmethod
    def _is_confident(cls, route) -> bool:
        """True when a route produced a real answer, not a soft miss and
        not a transient web error (translating won't fix a rate limit —
        it hits the same backend)."""
        tool = route.tool_used or ""
        return (
            route.direct_reply is not None
            and tool not in cls._SOFT_MISS_TOOLS
            and not tool.startswith("web_error")
        )

    def _resolve_route(self, conversation_id: str, user_message: str):
        """Route one message, with an additive Portuguese->English->
        Portuguese fallback. Returns (direct_reply, web_snippets,
        tool_used); direct_reply is None when nothing matched and the
        caller should generate (or give the no-freeform reply).

        The fallback fires only when the native pass produced no confident
        answer, so it can never degrade the things Aila already does well
        in Portuguese (greetings, identity, maths, a found memory,
        Portuguese Wikipedia). It exists to let a Portuguese question that
        Portuguese sources couldn't answer reach the far larger English
        Wikipedia — and to find an English-stored memory — then come back
        translated.
        """
        if self.router is None:
            return None, None, None

        prev = self._previous_user_message(conversation_id)
        route = self.router.route(user_message, previous_user_message=prev)
        if self._is_confident(route):
            return route.direct_reply, None, route.tool_used

        # The English retry is only worth its network cost on a *soft
        # miss* — the native pass reached memory or a search and came up
        # empty (memory_miss / web_no_answer). Anything else (a message
        # that matched no tool at all, a non-question, a transient web
        # error) cannot be helped by translating: the English pipeline
        # applies the same routing rules and hits the same backends. This
        # gate is what stops "Faça um poema" from paying for a pointless
        # translation call before falling through to the no-freeform
        # reply.
        soft_miss = (route.tool_used or "") in self._SOFT_MISS_TOOLS

        translator = self.translator
        if (
            soft_miss
            and translator is not None
            and getattr(translator, "available", False)
            and detect_language(user_message) == "pt"
        ):
            self._emit_status("Translating and searching in English...")
            english = translator.to_english(user_message)
            # Only proceed if translation actually produced something new;
            # if it passed through unchanged (offline, error), the native
            # route already covered it.
            if english and english.strip().lower() != user_message.strip().lower():
                prev_en = translator.to_english(prev) if prev else None
                route_en = self.router.route(english, previous_user_message=prev_en)
                if self._is_confident(route_en):
                    translated = translator.to_portuguese(route_en.direct_reply)
                    logger.info("turn path=translated:%s", route_en.tool_used)
                    return translated, None, f"translated:{route_en.tool_used}"

        # No confident answer anywhere. Fall back to whatever the native
        # route produced — a Portuguese soft-miss reply, injectable
        # snippets, or nothing.
        if route.direct_reply is not None:
            return route.direct_reply, None, route.tool_used
        return None, route.context_snippets or None, route.tool_used

    # -- inference -----------------------------------------------------

    @torch.no_grad()
    def respond(
        self,
        conversation_id: str,
        user_message: str,
        settings: GenerationSettings | None = None,
        remember_turn: bool = True,
    ) -> str:
        started = time.time()
        command_reply = self._handle_memory_command(user_message)
        if command_reply is not None:
            if remember_turn:
                self._remember_turn(conversation_id, user_message, command_reply)
            logger.info("turn path=memory_command latency=%.3fs", time.time() - started)
            return command_reply

        # Tool routing (calculator / knowledge base / web research) —
        # deterministic, runs before generation, never raises (see
        # tools/router.py). A direct reply skips the model entirely;
        # context snippets ride along into the system prompt as [WEB] data.
        direct_reply, web_snippets, tool_used = self._resolve_route(
            conversation_id, user_message
        )
        if direct_reply is not None:
            if remember_turn:
                self._remember_turn(conversation_id, user_message, direct_reply)
            logger.info("turn path=tool:%s latency=%.3fs", tool_used, time.time() - started)
            return direct_reply

        if not self.allow_freeform:
            reply = self._no_freeform_reply(user_message)
            if remember_turn:
                self._remember_turn(conversation_id, user_message, reply)
            logger.info("turn path=no_freeform latency=%.3fs", time.time() - started)
            return reply

        settings = settings or self.default_settings
        input_tensor, prompt_ids = self._prepare_turn(
            conversation_id, user_message, web_snippets=web_snippets
        )

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

        elapsed = time.time() - started
        logger.info(
            "turn path=generation tokens=%d latency=%.3fs (%.0f tok/s)%s",
            len(new_ids),
            elapsed,
            len(new_ids) / elapsed if elapsed > 0 else 0.0,
            " web_context=yes" if web_snippets else "",
        )
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

        direct_reply, web_snippets, _tool_used = self._resolve_route(
            conversation_id, user_message
        )
        if direct_reply is not None:
            yield direct_reply
            if remember_turn:
                self._remember_turn(conversation_id, user_message, direct_reply)
            return

        if not self.allow_freeform:
            reply = self._no_freeform_reply(user_message)
            yield reply
            if remember_turn:
                self._remember_turn(conversation_id, user_message, reply)
            return

        settings = settings or self.default_settings
        input_tensor, _ = self._prepare_turn(
            conversation_id, user_message, web_snippets=web_snippets
        )

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
