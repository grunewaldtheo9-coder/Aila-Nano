"""Aila Nano API — FastAPI backend serving chat (incl. streaming),
agents, memory, and file upload endpoints to the Next.js web frontend
(or any other client).

Run with:
    uvicorn web.backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web.backend.app.deps import AilaSettings, AilaState
from web.backend.app.routers import agents, chat, health, memory, upload

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading Aila Nano model, tokenizer, and memory stores...")
    app.state.aila = AilaState(AilaSettings())
    logger.info("Aila Nano API ready.")
    yield
    app.state.aila.close()


app = FastAPI(
    title="Aila Nano API",
    description="Backend API for Aila Nano, an original small language model by Aila Company Solutions.",
    version="0.1.0",
    lifespan=lifespan,
)

_allowed_origins = os.environ.get("AILA_CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(agents.router)
app.include_router(memory.router)
app.include_router(upload.router)
