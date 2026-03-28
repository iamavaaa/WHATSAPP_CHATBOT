"""
Sanity-check configuration before running the app or building the index.

Loads `.env` from the repository root (parent of `scripts/`). Does not print secrets.

On a host that only runs the API (no JSONL on disk), set VERIFY_SKIP_RAG_FILES=1.

Usage (from repo root):
  python scripts/verify_setup.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    for name in ("GOOGLE_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"):
        if not (os.getenv(name) or "").strip():
            failures.append(f"Missing or empty: {name}")

    skip_rag_files = (os.getenv("VERIFY_SKIP_RAG_FILES", "") or "").lower() in (
        "1",
        "true",
        "yes",
    )
    if skip_rag_files:
        warnings.append(
            "VERIFY_SKIP_RAG_FILES set: skipping JSONL file checks (use on deploy hosts without corpus files)"
        )
    else:
        rag_raw = (os.getenv("RAG_DATA_JSONL") or "").strip()
        if not rag_raw:
            warnings.append(
                "RAG_DATA_JSONL unset (build script will default to data/commando_networks.jsonl)"
            )
            rag_raw = str(ROOT / "data" / "commando_networks.jsonl")

        for part in rag_raw.split(","):
            part = part.strip()
            if not part:
                continue
            p = Path(part)
            path = p if p.is_absolute() else (ROOT / p)
            if not path.is_file():
                failures.append(f"RAG JSONL not found: {path}")

    use_pc = (os.getenv("USE_PINECONE", "true") or "").lower() == "true"
    if use_pc:
        key = (os.getenv("PINECONE_API_KEY") or "").strip()
        idx_name = (os.getenv("PINECONE_INDEX_NAME") or "").strip()
        if key and idx_name:
            try:
                from pinecone import Pinecone

                pc = Pinecone(api_key=key)
                raw = pc.list_indexes()
                names: set[str] = set()
                if hasattr(raw, "names"):
                    names = set(raw.names())
                elif isinstance(raw, list):
                    for item in raw:
                        if isinstance(item, str):
                            names.add(item)
                        elif isinstance(item, dict) and item.get("name"):
                            names.add(str(item["name"]))
                        elif hasattr(item, "name"):
                            names.add(str(item.name))
                if idx_name not in names:
                    warnings.append(
                        f"Pinecone index '{idx_name}' not found in project. Create it in the console "
                        "or run: cd scripts && python build_pinecone_index.py"
                    )
                else:
                    stats = pc.Index(idx_name).describe_index_stats()
                    total = 0
                    dimension = None
                    metric = None
                    if isinstance(stats, dict):
                        total = int(stats.get("total_vector_count") or 0)
                        dimension = stats.get("dimension")
                        metric = stats.get("metric")
                    else:
                        total = int(getattr(stats, "total_vector_count", 0) or 0)
                        dimension = getattr(stats, "dimension", None)
                        metric = getattr(stats, "metric", None)
                    ns = os.getenv("PINECONE_NAMESPACE", "") or "(default)"
                    dim_s = str(dimension) if dimension is not None else "?"
                    met_s = str(metric) if metric is not None else "?"
                    print(
                        f"[INFO] Pinecone index '{idx_name}' (namespace={ns!r}): "
                        f"total_vector_count={total}, dimension={dim_s}, metric={met_s}"
                    )
                    if total == 0:
                        warnings.append(
                            "Pinecone reports 0 vectors for this index. If the console shows records here "
                            "but this says 0, your PINECONE_API_KEY likely belongs to a different Pinecone "
                            "project than the browser. If both are 0, run: python scripts/build_pinecone_index.py"
                        )
            except Exception as exc:
                warnings.append(f"Pinecone API check skipped/failed: {exc!s}")

    for w in warnings:
        print(f"[WARN] {w}")
    for f in failures:
        print(f"[FAIL] {f}")

    if not failures:
        print("[OK] Required env vars and RAG JSONL paths look good.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
