"""Chat endpoints: a plain request/response route and a streaming (SSE)
route the web frontend uses to render tokens as they're generated.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from agents.base import GenerationSettings
from web.backend.app.deps import AilaState, get_state
from web.backend.app.schemas import ChatRequest, ChatResponse, GenerationSettingsSchema

router = APIRouter(prefix="/chat", tags=["chat"])


def _resolve_settings(schema: GenerationSettingsSchema | None) -> GenerationSettings | None:
    if schema is None:
        return None
    return GenerationSettings(**schema.model_dump())


def _get_agent_or_404(state: AilaState, agent_name: str):
    if agent_name not in state.available_agents():
        raise HTTPException(status_code=404, detail=f"Unknown agent '{agent_name}'")
    return state.get_agent(agent_name)


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, state: AilaState = Depends(get_state)) -> ChatResponse:
    agent = _get_agent_or_404(state, req.agent)
    reply = agent.respond(
        req.conversation_id,
        req.message,
        settings=_resolve_settings(req.settings),
        remember_turn=req.remember_turn,
    )
    return ChatResponse(conversation_id=req.conversation_id, agent=agent.name, reply=reply)


@router.post("/stream")
def chat_stream(req: ChatRequest, state: AilaState = Depends(get_state)):
    agent = _get_agent_or_404(state, req.agent)
    settings = _resolve_settings(req.settings)

    def event_generator():
        try:
            for delta in agent.respond_stream(
                req.conversation_id, req.message, settings=settings, remember_turn=req.remember_turn
            ):
                yield {"event": "token", "data": json.dumps({"delta": delta})}
            yield {"event": "done", "data": json.dumps({"conversation_id": req.conversation_id})}
        except Exception as e:  # surface generation errors to the client stream
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_generator())
