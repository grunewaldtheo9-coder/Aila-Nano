import pytest
from fastapi.testclient import TestClient

from tests.conftest import SAMPLE_FINETUNE


@pytest.fixture
def client(tokenizer, tmp_path, monkeypatch):
    monkeypatch.setenv("AILA_CHECKPOINT", str(tmp_path / "does-not-exist.pt"))
    monkeypatch.setenv("AILA_FALLBACK_CHECKPOINT", str(tmp_path / "also-missing.pt"))
    monkeypatch.setenv("AILA_TOKENIZER", tokenizer.model_path)
    monkeypatch.setenv("AILA_MEMORY_DB", str(tmp_path / "mem.db"))
    monkeypatch.setenv("AILA_MEMORY_FAISS", str(tmp_path / "mem.faiss"))
    monkeypatch.setenv("AILA_KNOWLEDGE_DB", str(tmp_path / "kb.db"))
    monkeypatch.setenv("AILA_KNOWLEDGE_FAISS", str(tmp_path / "kb.faiss"))
    monkeypatch.setenv("AILA_DEVICE", "cpu")

    # Import (or re-import) after env vars are set so a freshly-constructed
    # AilaState in the lifespan handler picks them up.
    import importlib

    from web.backend.app import main as main_module

    importlib.reload(main_module)

    with TestClient(main_module.app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False  # no checkpoint exists in this fixture
    assert body["vocab_size"] == 384
    assert set(body["agents"]) == {"general", "programming", "research", "writing"}


def test_list_agents_endpoint(client):
    r = client.get("/agents")
    assert r.status_code == 200
    names = {a["name"] for a in r.json()}
    assert names == {"general", "programming", "research", "writing"}


def test_unknown_agent_returns_404(client):
    r = client.get("/agents/not-real")
    assert r.status_code == 404


def test_chat_roundtrip_and_history(client):
    r = client.post(
        "/chat", json={"conversation_id": "t1", "message": "Hello!", "agent": "general"}
    )
    assert r.status_code == 200
    assert "reply" in r.json()

    r = client.get("/memory/conversations/t1")
    assert r.status_code == 200
    assert len(r.json()["messages"]) == 2


def test_chat_unknown_agent_returns_404(client):
    r = client.post(
        "/chat", json={"conversation_id": "t1", "message": "hi", "agent": "nonexistent"}
    )
    assert r.status_code == 404


def test_remember_and_recall(client):
    r = client.post("/memory/remember", json={"content": "The user likes tea.", "importance": 0.7})
    assert r.status_code == 200
    assert "id" in r.json()

    r = client.get("/memory/recall", params={"query": "What drink does the user like?"})
    assert r.status_code == 200
    assert len(r.json()["results"]) >= 1


def test_upload_indexes_document(client):
    content = SAMPLE_FINETUNE.read_bytes()
    r = client.post("/upload", files={"file": ("data.jsonl", content, "application/json")})
    assert r.status_code == 200
    body = r.json()
    assert body["chunks_indexed"] >= 1


def test_upload_rejects_disallowed_extension(client):
    r = client.post("/upload", files={"file": ("evil.exe", b"binary", "application/octet-stream")})
    assert r.status_code == 400


def test_clear_conversation(client):
    client.post("/chat", json={"conversation_id": "t2", "message": "hi", "agent": "general"})
    r = client.delete("/memory/conversations/t2")
    assert r.status_code == 200
    r = client.get("/memory/conversations/t2")
    assert r.json()["messages"] == []
