"""Content identities for the actual evaluated records and executing source tree."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from retrieval_bench.models import Document, RetrievalQuery


def _digest(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_provenance(
    documents: list[Document],
    queries: list[RetrievalQuery],
    *,
    model_name: str,
    model_revision: str | None,
) -> dict[str, Any]:
    """Hash parsed records in execution order, not possibly stale input filenames."""
    source_root = Path(__file__).resolve().parents[2]
    commit = None
    dirty = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=source_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        pass  # Installed wheels may have no source checkout. Record unknown, not invented data.
    return {
        "hash_scope": "canonical parsed records in execution order; UTF-8 JSON",
        "corpus_sha256": _digest([record.model_dump() for record in documents]),
        "queries_sha256": _digest([record.model_dump() for record in queries]),
        "document_count": len(documents),
        "query_count": len(queries),
        "model_name": model_name,
        "model_revision": model_revision,
        "model_revision_pinned": model_revision is not None,
        "git_commit": commit,
        "git_dirty": dirty,
    }
