# Aila Nano — backend/training image.
#
# CPU by default. For CUDA, change the base image to e.g.
# `nvidia/cuda:12.1.0-runtime-ubuntu22.04`, install Python there, and swap
# the torch install line below for the cu121 wheel index (see
# docs/INSTALL.md).
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch>=2.2 \
    && pip install -r requirements.txt

COPY . .

EXPOSE 8000

# Model/tokenizer/memory paths are configurable via env vars — see
# web/backend/app/deps.py — and are typically bind-mounted in (see
# docker-compose.yml) so checkpoints trained outside the container are
# visible to it.
CMD ["uvicorn", "web.backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
