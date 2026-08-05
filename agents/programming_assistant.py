from agents.base import AILA_KNOWLEDGE_PRIMER, Agent, GenerationSettings


class ProgrammingAssistant(Agent):
    name = "programming"
    system_prompt = (
        AILA_KNOWLEDGE_PRIMER
        + "You are a precise, pragmatic programming assistant. Prefer correct, minimal code "
        "over verbose explanations, use idiomatic style for the language in question, and call "
        "out edge cases or assumptions briefly when they matter."
    )
    # Lower temperature: code correctness benefits more from determinism
    # than creative-writing-style variety.
    default_settings = GenerationSettings(temperature=0.4, top_k=30, top_p=0.9)
