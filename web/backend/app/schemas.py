"""Pydantic request/response models for the Aila Nano API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerationSettingsSchema(BaseModel):
    max_new_tokens: int = Field(default=200, ge=1, le=2000)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int | None = Field(default=40, ge=1)
    top_p: float | None = Field(default=0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.15, ge=1.0, le=2.0)


class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    agent: str = "general"
    settings: GenerationSettingsSchema | None = None
    remember_turn: bool = True


class ChatResponse(BaseModel):
    conversation_id: str
    agent: str
    reply: str


class AgentInfo(BaseModel):
    name: str
    system_prompt: str


class MessageOut(BaseModel):
    role: str
    content: str
    agent_type: str | None = None
    created_at: float


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    messages: list[MessageOut]


class RememberFactRequest(BaseModel):
    content: str
    importance: float = Field(default=0.6, ge=0.0, le=1.0)


class RememberFactResponse(BaseModel):
    id: int


class RecallResult(BaseModel):
    id: int
    content: str
    score: float
    combined_score: float
    importance: float
    created_at: float


class RecallResponse(BaseModel):
    query: str
    results: list[RecallResult]


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    document_ids: list[int]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_parameters: int | None = None
    vocab_size: int | None = None
    device: str
    agents: list[str]
