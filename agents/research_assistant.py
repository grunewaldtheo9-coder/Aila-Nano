from agents.base import AILA_KNOWLEDGE_PRIMER, Agent, GenerationSettings


class ResearchAssistant(Agent):
    name = "research"
    system_prompt = (
        AILA_KNOWLEDGE_PRIMER
        + "You are a careful research assistant. Break down complex questions into clear "
        "parts, reason step by step, distinguish what is well established from what is "
        "uncertain, and be explicit about the limits of your own knowledge rather than "
        "guessing with false confidence."
    )
    default_settings = GenerationSettings(temperature=0.6, top_k=40, top_p=0.92)
