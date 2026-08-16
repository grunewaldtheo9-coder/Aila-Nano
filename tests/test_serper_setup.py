"""First-run Serper setup: the .env writer, the key check, attaching the
source at runtime, and the terminal prompt's guards.

The prompt itself writes a secret to disk, so the rules it must obey are
tested rather than assumed: never write outside a .env file, never
consume input from a non-terminal, never save a key that doesn't work,
and never print the key.
"""

from __future__ import annotations

import os

import pytest

from engine.env import load_env, save_env_var

FAKE_KEY = "test-key-0123456789abcdef"


# -- .env writing -------------------------------------------------------------


def test_save_env_var_creates_the_file_and_round_trips(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    monkeypatch.delenv("SERPER_API_KEY", raising=False)

    save_env_var("SERPER_API_KEY", FAKE_KEY, path=env_path)
    assert env_path.exists()
    assert f"SERPER_API_KEY={FAKE_KEY}" in env_path.read_text(encoding="utf-8")

    # It is live immediately, and survives a reload.
    assert os.environ["SERPER_API_KEY"] == FAKE_KEY
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    load_env(env_path)
    assert os.environ["SERPER_API_KEY"] == FAKE_KEY


def test_save_env_var_replaces_only_its_own_line(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# a comment\nAILA_DEVICE=cpu\nSERPER_API_KEY=old-key\nAILA_DAILY_STUDY=false\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SERPER_API_KEY", raising=False)

    save_env_var("SERPER_API_KEY", "new-key", path=env_path)
    text = env_path.read_text(encoding="utf-8")
    assert "SERPER_API_KEY=new-key" in text
    assert "old-key" not in text
    # Everything else is preserved, comments included.
    assert "# a comment" in text
    assert "AILA_DEVICE=cpu" in text
    assert "AILA_DAILY_STUDY=false" in text


def test_save_env_var_is_owner_only(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    save_env_var("SERPER_API_KEY", FAKE_KEY, path=env_path)
    mode = env_path.stat().st_mode & 0o777
    # Windows ignores chmod; skip the assertion there rather than fail.
    if os.name == "posix":
        assert mode == 0o600, f"secret file is {oct(mode)}, expected 0o600"


def test_save_env_var_refuses_to_write_secrets_anywhere_else(tmp_path):
    """The only thing keeping the key out of the repository is that .env
    is gitignored. Writing a secret into any other filename would defeat
    that, so the function refuses."""
    for name in ("config.py", "settings.txt", "README.md", "env"):
        with pytest.raises(ValueError, match="expected a .env file"):
            save_env_var("SERPER_API_KEY", FAKE_KEY, path=tmp_path / name)

    # ".env.local" and friends are fine.
    save_env_var("SERPER_API_KEY", FAKE_KEY, path=tmp_path / ".env.local")


def test_save_env_var_rejects_malformed_input(tmp_path):
    with pytest.raises(ValueError):
        save_env_var("BAD KEY NAME", FAKE_KEY, path=tmp_path / ".env")
    with pytest.raises(ValueError):
        # A newline would let one value forge additional variables.
        save_env_var("SERPER_API_KEY", "abc\nAILA_DEVICE=cuda", path=tmp_path / ".env")


# -- engine: checking and attaching a key -------------------------------------


def _engine(tmp_path, monkeypatch, **extra):
    from engine import AilaEngine, EngineSettings

    env = {
        "AILA_CHECKPOINT": str(tmp_path / "missing.pt"),
        "AILA_FALLBACK_CHECKPOINT": str(tmp_path / "missing2.pt"),
        "AILA_MEMORY_DB": str(tmp_path / "m.db"),
        "AILA_MEMORY_FAISS": str(tmp_path / "m.faiss"),
        "AILA_KNOWLEDGE_DB": str(tmp_path / "k.db"),
        "AILA_KNOWLEDGE_FAISS": str(tmp_path / "k.faiss"),
        "AILA_KNOWLEDGE_STORE_DB": str(tmp_path / "ks.db"),
        "AILA_DEVICE": "cpu",
        "AILA_WIKIPEDIA_ENABLED": "true",  # a pipeline must exist to attach to
        "SERPER_API_KEY": "",
        **extra,
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return AilaEngine(EngineSettings())


def test_a_key_can_be_attached_without_restarting(tokenizer, tmp_path, monkeypatch):
    monkeypatch.setenv("AILA_TOKENIZER", tokenizer.model_path)
    with _engine(tmp_path, monkeypatch) as engine:
        assert engine.research_sources == ["wikipedia"]
        assert engine.set_serper_api_key(FAKE_KEY) is True
        assert engine.research_sources == ["wikipedia", "serper"]
        # Wikipedia stays first: free, no quota, complete sentences.
        assert engine.research_sources[0] == "wikipedia"


def test_attaching_an_empty_key_is_a_no_op(tokenizer, tmp_path, monkeypatch):
    monkeypatch.setenv("AILA_TOKENIZER", tokenizer.model_path)
    with _engine(tmp_path, monkeypatch) as engine:
        assert engine.set_serper_api_key("   ") is False
        assert engine.research_sources == ["wikipedia"]


def test_key_check_reports_a_rejected_key(tokenizer, tmp_path, monkeypatch):
    monkeypatch.setenv("AILA_TOKENIZER", tokenizer.model_path)
    import urllib.error
    import urllib.request

    def rejected(request, timeout=None):
        raise urllib.error.HTTPError("https://google.serper.dev/search", 403, "Forbidden", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", rejected)
    with _engine(tmp_path, monkeypatch) as engine:
        ok, reason = engine.check_serper_api_key(FAKE_KEY)
        assert ok is False
        assert "rejected" in reason
        assert FAKE_KEY not in reason  # never echo the key


def test_key_check_treats_a_rate_limit_as_a_working_key(tokenizer, tmp_path, monkeypatch):
    """A 429 means the key is real and simply out of searches. Refusing
    to save it would be wrong."""
    monkeypatch.setenv("AILA_TOKENIZER", tokenizer.model_path)
    import urllib.error
    import urllib.request

    def limited(request, timeout=None):
        raise urllib.error.HTTPError("https://google.serper.dev/search", 429, "Too Many", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", limited)
    with _engine(tmp_path, monkeypatch) as engine:
        ok, reason = engine.check_serper_api_key(FAKE_KEY)
        assert ok is True
        assert "limit" in reason


def test_key_check_rejects_an_empty_key(tokenizer, tmp_path, monkeypatch):
    monkeypatch.setenv("AILA_TOKENIZER", tokenizer.model_path)
    with _engine(tmp_path, monkeypatch) as engine:
        assert engine.check_serper_api_key("") == (False, "No key entered.")


# -- the terminal prompt's guards ---------------------------------------------


def test_setup_never_consumes_piped_input(tokenizer, tmp_path, monkeypatch, capsys):
    """The prompt reads a line from stdin. In a scripted or piped run
    that line would be the user's first message, silently eaten and
    treated as an API key."""
    import chat

    monkeypatch.setenv("AILA_TOKENIZER", tokenizer.model_path)
    with _engine(tmp_path, monkeypatch) as engine:
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        chat.offer_serper_setup(engine)
        assert capsys.readouterr().out == ""


def test_setup_is_skipped_when_a_key_is_already_configured(tokenizer, tmp_path, monkeypatch, capsys):
    import chat

    monkeypatch.setenv("AILA_TOKENIZER", tokenizer.model_path)
    with _engine(tmp_path, monkeypatch, SERPER_API_KEY=FAKE_KEY) as engine:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        chat.offer_serper_setup(engine)
        assert capsys.readouterr().out == ""


def test_setup_is_asked_once_and_remembers_being_declined(tokenizer, tmp_path, monkeypatch, capsys):
    import chat

    monkeypatch.setenv("AILA_TOKENIZER", tokenizer.model_path)
    with _engine(tmp_path, monkeypatch) as engine:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("getpass.getpass", lambda prompt="": "")  # user presses Enter

        chat.offer_serper_setup(engine)
        first = capsys.readouterr().out
        assert "For better researching" in first
        assert "I'll use Wikipedia" in first

        # Second start: no nagging.
        chat.offer_serper_setup(engine)
        assert capsys.readouterr().out == ""

        # ...but /serper asks again on demand.
        chat.offer_serper_setup(engine, force=True)
        assert "For better researching" in capsys.readouterr().out


def test_setup_does_not_save_a_key_that_does_not_work(tokenizer, tmp_path, monkeypatch, capsys):
    import chat

    monkeypatch.setenv("AILA_TOKENIZER", tokenizer.model_path)
    with _engine(tmp_path, monkeypatch) as engine:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("getpass.getpass", lambda prompt="": "a-dead-key")
        monkeypatch.setattr(
            engine, "check_serper_api_key", lambda key: (False, "That key was rejected.")
        )
        saved: list = []
        monkeypatch.setattr(chat, "save_env_var", lambda *a, **k: saved.append(a))

        chat.offer_serper_setup(engine)
        out = capsys.readouterr().out
        assert "Nothing was saved" in out
        assert saved == []
        assert engine.research_sources == ["wikipedia"]
        assert "a-dead-key" not in out  # never echo the key


def test_setup_saves_and_uses_a_working_key(tokenizer, tmp_path, monkeypatch, capsys):
    import chat

    monkeypatch.setenv("AILA_TOKENIZER", tokenizer.model_path)
    with _engine(tmp_path, monkeypatch) as engine:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("getpass.getpass", lambda prompt="": FAKE_KEY)
        monkeypatch.setattr(engine, "check_serper_api_key", lambda key: (True, "The key works."))
        saved: list = []
        monkeypatch.setattr(chat, "save_env_var", lambda *a, **k: saved.append(a))

        chat.offer_serper_setup(engine)
        out = capsys.readouterr().out
        assert "Saved to your .env file" in out
        assert saved == [("SERPER_API_KEY", FAKE_KEY)]
        # Usable immediately, no restart.
        assert engine.research_sources == ["wikipedia", "serper"]
        assert FAKE_KEY not in out  # never echo the key


def test_setup_survives_an_unwritable_env_file(tokenizer, tmp_path, monkeypatch, capsys):
    import chat

    monkeypatch.setenv("AILA_TOKENIZER", tokenizer.model_path)
    with _engine(tmp_path, monkeypatch) as engine:
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("getpass.getpass", lambda prompt="": FAKE_KEY)
        monkeypatch.setattr(engine, "check_serper_api_key", lambda key: (True, "The key works."))

        def unwritable(*a, **k):
            raise OSError("read-only file system")

        monkeypatch.setattr(chat, "save_env_var", unwritable)

        chat.offer_serper_setup(engine)
        out = capsys.readouterr().out
        assert "this session only" in out
        # Still attached for this run rather than thrown away.
        assert engine.research_sources == ["wikipedia", "serper"]
