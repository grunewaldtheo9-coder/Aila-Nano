from fastapi import APIRouter, Depends, HTTPException

from web.backend.app.deps import AilaState, get_state
from web.backend.app.schemas import AgentInfo

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("", response_model=list[AgentInfo])
def list_agents_endpoint(state: AilaState = Depends(get_state)) -> list[AgentInfo]:
    infos = []
    for name in state.available_agents():
        agent = state.get_agent(name)
        infos.append(AgentInfo(name=agent.name, system_prompt=agent.system_prompt))
    return infos


@router.get("/{agent_name}", response_model=AgentInfo)
def get_agent_endpoint(agent_name: str, state: AilaState = Depends(get_state)) -> AgentInfo:
    if agent_name not in state.available_agents():
        raise HTTPException(status_code=404, detail=f"Unknown agent '{agent_name}'")
    agent = state.get_agent(agent_name)
    return AgentInfo(name=agent.name, system_prompt=agent.system_prompt)
