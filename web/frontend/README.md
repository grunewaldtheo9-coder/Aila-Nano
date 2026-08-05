# Aila Nano — Web Interface

A Next.js (App Router, TypeScript, Tailwind CSS) chat UI for Aila Nano.

## Features

- Streaming chat (SSE) with the Aila Nano FastAPI backend
- Four agent personas (General / Programming / Research / Writing)
- Dark mode with system-preference detection and persistence
- Conversation history sidebar, backed by the backend's memory API
- File upload into Aila Nano's semantic knowledge base
- Adjustable generation settings (temperature, top-k/p, repetition penalty)
- Responsive layout (mobile sidebar drawer)

## Setup

```bash
cd web/frontend
npm install
cp .env.local.example .env.local   # point NEXT_PUBLIC_API_URL at the backend
npm run dev
```

The backend (see `web/backend/`) must be running for the UI to do anything
useful — start it first with `uvicorn web.backend.app.main:app --reload`.

## Build

```bash
npm run build && npm run start
```
