from agents.base import AILA_KNOWLEDGE_PRIMER, Agent, GenerationSettings


class WritingAssistant(Agent):
    name = "writing"
    system_prompt = (
        AILA_KNOWLEDGE_PRIMER
        + "You are a skilled writing assistant. Adapt tone and style to what the user asks "
        "for, favor clear and vivid language over filler, and offer light editorial "
        "suggestions when they would genuinely improve the piece."
    )
    # Higher temperature: creative writing benefits from more variety.
    default_settings = GenerationSettings(temperature=0.95, top_k=50, top_p=0.95)
