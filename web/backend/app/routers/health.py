from fastapi import APIRouter, Depends

from model.utils import count_parameters
from web.backend.app.deps import AilaState, get_state
from web.backend.app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(state: AilaState = Depends(get_state)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=state.model_loaded_from is not None,
        num_parameters=count_parameters(state.model),
        vocab_size=state.tokenizer.vocab_size,
        device=state.device,
        agents=state.available_agents(),
    )
