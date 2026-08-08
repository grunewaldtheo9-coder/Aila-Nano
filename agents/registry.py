"""Central registry of available agent personas, so any interface (the
terminal chat, or a future GUI/mobile/web front end) can look one up by
name without importing every class directly.
"""

from __future__ import annotations

from agents.base import Agent
from agents.general_assistant import GeneralAssistant
from agents.programming_assistant import ProgrammingAssistant
from agents.research_assistant import ResearchAssistant
from agents.writing_assistant import WritingAssistant
from memory.manager import MemoryManager
from model.transformer import AilaNanoGPT
from tokenizer.tokenizer import AilaTokenizer
from vectordb.semantic_index import SemanticIndex

AGENT_REGISTRY: dict[str, type[Agent]] = {
    GeneralAssistant.name: GeneralAssistant,
    ProgrammingAssistant.name: ProgrammingAssistant,
    ResearchAssistant.name: ResearchAssistant,
    WritingAssistant.name: WritingAssistant,
}


def list_agents() -> list[str]:
    return list(AGENT_REGISTRY.keys())


def get_agent(
    agent_name: str,
    model: AilaNanoGPT,
    tokenizer: AilaTokenizer,
    memory: MemoryManager | None = None,
    knowledge: SemanticIndex | None = None,
    device: str = "cpu",
    router=None,
) -> Agent:
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent '{agent_name}'. Available agents: {', '.join(list_agents())}"
        )
    agent_cls = AGENT_REGISTRY[agent_name]
    return agent_cls(
        model, tokenizer, memory=memory, knowledge=knowledge, device=device, router=router
    )
