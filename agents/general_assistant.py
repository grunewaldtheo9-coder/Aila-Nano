from agents.base import AILA_KNOWLEDGE_PRIMER, Agent, GenerationSettings


class GeneralAssistant(Agent):
    name = "general"
    system_prompt = (
        AILA_KNOWLEDGE_PRIMER
        + "You are a friendly, knowledgeable general-purpose assistant. Answer clearly and "
        "concisely, and ask a clarifying question when the user's request is ambiguous."
    )
    default_settings = GenerationSettings(temperature=0.8, top_k=40, top_p=0.95)
