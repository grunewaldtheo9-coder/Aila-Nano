from fastapi import APIRouter, Depends

from web.backend.app.deps import AilaState, get_state
from web.backend.app.schemas import (
    ConversationHistoryResponse,
    MessageOut,
    RecallResponse,
    RecallResult,
    RememberFactRequest,
    RememberFactResponse,
)

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/conversations", response_model=list[str])
def list_conversations(state: AilaState = Depends(get_state)) -> list[str]:
    return state.memory.conversation.list_conversations()


@router.get("/conversations/{conversation_id}", response_model=ConversationHistoryResponse)
def get_history(
    conversation_id: str, state: AilaState = Depends(get_state)
) -> ConversationHistoryResponse:
    history = state.memory.conversation.get_history(conversation_id, max_turns=None)
    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        messages=[
            MessageOut(
                role=m["role"],
                content=m["content"],
                agent_type=m.get("agent_type"),
                created_at=m["created_at"],
            )
            for m in history
        ],
    )


@router.delete("/conversations/{conversation_id}")
def clear_history(conversation_id: str, state: AilaState = Depends(get_state)) -> dict:
    state.memory.conversation.clear(conversation_id)
    return {"conversation_id": conversation_id, "cleared": True}


@router.post("/remember", response_model=RememberFactResponse)
def remember_fact(
    req: RememberFactRequest, state: AilaState = Depends(get_state)
) -> RememberFactResponse:
    fact_id = state.memory.remember_fact(req.content, importance=req.importance)
    return RememberFactResponse(id=fact_id)


@router.get("/recall", response_model=RecallResponse)
def recall(query: str, k: int = 5, state: AilaState = Depends(get_state)) -> RecallResponse:
    results = state.memory.semantic.recall(query, k=k)
    return RecallResponse(
        query=query,
        results=[
            RecallResult(
                id=r["id"],
                content=r["content"],
                score=r["score"],
                combined_score=r["combined_score"],
                importance=r["importance"],
                created_at=r["created_at"],
            )
            for r in results
        ],
    )
