"""Dataset versioning: a reproducible manifest that records exactly what a
tokenized corpus is, so a training checkpoint can point at the dataset that
produced it (spec: dataset versioning).

A manifest is a JSON document stored at
`datasets/pretrain/<version>/manifest.json` containing: version, creation
timestamp, git commit, tokenizer path + vocab size, the corpus content hash
and token counts (train/val), the source list with license/provenance, the
language distribution, document/token counts, and the filtering
configuration used. Nothing here is fabricated — sources and licenses are
whatever the caller passes, and the hashes are computed from the real files.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from training.dataset import corpus_fingerprint


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


@dataclass
class Source:
    name: str
    license: str
    url: str = ""
    provenance: str = ""  # how it was obtained / verified
    documents: int | None = None
    tokens: int | None = None


@dataclass
class DatasetManifest:
    version: str
    tokenizer_path: str
    tokenizer_vocab_size: int
    train_bin: str
    val_bin: str = ""
    train_sha256: str = ""
    train_tokens: int = 0
    val_sha256: str = ""
    val_tokens: int = 0
    document_count: int = 0
    language_distribution: dict = field(default_factory=dict)  # docs by language
    token_distribution: dict = field(default_factory=dict)     # tokens by language/source
    filter_config: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    created_at: str = ""
    git_commit: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def build_manifest(
    version: str,
    train_bin: str,
    tokenizer_path: str,
    tokenizer_vocab_size: int,
    val_bin: str = "",
    sources: list[Source] | None = None,
    language_distribution: dict | None = None,
    token_distribution: dict | None = None,
    filter_config: dict | None = None,
    document_count: int = 0,
) -> DatasetManifest:
    """Assemble a manifest from real files: the corpus hashes and token
    counts are computed by streaming the .bin(s); everything else is
    recorded as provided by the caller."""
    train_fp = corpus_fingerprint(train_bin)
    m = DatasetManifest(
        version=version,
        tokenizer_path=tokenizer_path,
        tokenizer_vocab_size=tokenizer_vocab_size,
        train_bin=train_bin,
        val_bin=val_bin,
        train_sha256=train_fp["sha256"],
        train_tokens=train_fp["num_tokens"],
        document_count=document_count,
        language_distribution=language_distribution or {},
        token_distribution=token_distribution or {},
        filter_config=filter_config or {},
        sources=[asdict(s) if isinstance(s, Source) else s for s in (sources or [])],
        created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=_git_commit(),
    )
    if val_bin and Path(val_bin).exists():
        val_fp = corpus_fingerprint(val_bin)
        m.val_sha256 = val_fp["sha256"]
        m.val_tokens = val_fp["num_tokens"]
    return m


def write_manifest(manifest: DatasetManifest, root: str = "datasets/pretrain") -> Path:
    out_dir = Path(root) / manifest.version
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(manifest.to_json())
    return path


def load_manifest(version: str, root: str = "datasets/pretrain") -> dict:
    return json.loads((Path(root) / version / "manifest.json").read_text())
