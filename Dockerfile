# Aila Nano — terminal image.
#
# CPU by default. For CUDA, change the base image to e.g.
# `nvidia/cuda:12.1.0-runtime-ubuntu22.04`, install Python there, and swap
# the torch install line below for the cuXXX wheel index (see
# docs/INSTALL.md).
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6" \
    && pip install -r requirements.txt

COPY . .

# Model/tokenizer/memory paths are configurable via env vars — see
# engine/config.py — and are typically bind-mounted in (e.g.
# `-v ./checkpoints:/app/checkpoints`) so a model trained outside the
# container is visible to it.
ENTRYPOINT ["python", "chat.py"]
