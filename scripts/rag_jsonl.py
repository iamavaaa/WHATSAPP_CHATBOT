"""Shared helpers for loading one or more JSONL corpora (RAG index build)."""

from __future__ import annotations

import json
import os
from pathlib import Path


def rag_data_jsonl_paths(base_dir: Path) -> list[Path]:
    """Resolve RAG_DATA_JSONL: comma-separated paths, relative to repo root unless absolute."""
    default = str(base_dir / "data" / "sample_common_crawl.jsonl")
    raw = os.getenv("RAG_DATA_JSONL", default)
    paths: list[Path] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        p = Path(part).expanduser()
        paths.append(p.resolve() if p.is_absolute() else (base_dir / p).resolve())
    return paths


def load_texts_from_jsonl_files(paths: list[Path]) -> list[str]:
    docs: list[str] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    text = (payload.get("text") or "").strip()
                    if text:
                        docs.append(text)
                except json.JSONDecodeError:
                    continue
    return docs
