"""Tests for datasets/scripts/ — clean_text.py and dedupe.py directly,
plus a regression test for a real bug: our own datasets/ directory has
the same name as the (optional) Hugging Face `datasets` PyPI package.
Python's import system always prefers an installed *regular* package
(one with __init__.py, like HF's `datasets`) over a same-named local
*namespace* package (ours — datasets/ deliberately has no __init__.py,
since it's a data directory, not a Python package), no matter what's
prepended to sys.path.

That means `import datasets.scripts...` anywhere in this codebase is a
landmine: it works fine when the optional `datasets` PyPI package (see
requirements-datasets.txt) isn't installed, and silently resolves to the
*wrong* package — raising `ModuleNotFoundError: No module named
'datasets.scripts'` — the moment it is. `download_pretrain_data.py` hit
this for real; this test file deliberately imports the same way that
script now does (bare module names via sys.path, never the `datasets.`
dotted prefix) so it keeps passing regardless of which extras are
installed, and so nothing reintroduces the dotted form.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "datasets" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from clean_text import clean_corpus, clean_document, is_low_quality  # noqa: E402
from dedupe import dedupe_corpus, exact_dedupe, near_dedupe  # noqa: E402


def test_clean_document_normalizes_whitespace():
    assert clean_document("Hello   world.\n\n\n\nBye.") == "Hello world.\n\nBye."


def test_is_low_quality_flags_short_and_symbol_heavy_text():
    assert is_low_quality("hi", min_chars=20)
    assert is_low_quality("1234567890!@#$%^&*()_+{}[]", min_alpha_ratio=0.9)
    assert not is_low_quality("This is a perfectly normal sentence of real words.")


def test_clean_corpus_drops_low_quality_documents():
    docs = ["A proper sentence with real words in it.", "###!!!", ""]
    cleaned = clean_corpus(docs, min_chars=10)
    assert cleaned == ["A proper sentence with real words in it."]


def test_exact_dedupe_removes_identical_documents():
    docs = ["Same text.", "Same text.", "Different text."]
    assert exact_dedupe(docs) == ["Same text.", "Different text."]


def test_near_dedupe_removes_near_identical_documents():
    docs = [
        "The quick brown fox jumps over the lazy dog today.",
        "The quick brown fox jumps over the lazy dog now.",
        "A completely unrelated sentence about something else entirely.",
    ]
    deduped = near_dedupe(docs, threshold=0.7)
    assert len(deduped) == 2


def test_dedupe_corpus_runs_both_passes():
    docs = ["Hello world.", "Hello world.", "Something else."]
    assert dedupe_corpus(docs) == ["Hello world.", "Something else."]


def test_download_script_imports_cleanly_even_with_hf_datasets_installed():
    """Regression test: importing download_pretrain_data.py must not
    raise ModuleNotFoundError, whether or not the real `datasets` PyPI
    package happens to be installed alongside it.
    """
    pytest.importorskip("datasets")  # only meaningful with it installed

    import download_pretrain_data

    importlib.reload(download_pretrain_data)  # ensure a fresh import, not a stale cache
    assert download_pretrain_data.clean_corpus is clean_corpus
    assert download_pretrain_data.dedupe_corpus is dedupe_corpus
