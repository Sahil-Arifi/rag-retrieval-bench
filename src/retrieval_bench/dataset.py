"""Strict JSONL dataset loading and cross-file validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from retrieval_bench.models import Document, RetrievalQuery


class DatasetValidationError(ValueError):
    """Raised when corpus or query data violates the benchmark schema."""


ModelT = TypeVar("ModelT", bound=BaseModel)


def _load_jsonl(path: Path, model_type: type[ModelT], record_label: str) -> list[ModelT]:
    records: list[ModelT] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetValidationError(
                        f"{path}:{line_number} contains invalid JSON: {exc.msg}"
                    ) from exc
                try:
                    records.append(model_type.model_validate(payload))
                except ValidationError as exc:
                    raise DatasetValidationError(
                        f"{path}:{line_number} contains an invalid {record_label}: {exc}"
                    ) from exc
    except OSError as exc:
        raise DatasetValidationError(f"could not read {path}: {exc}") from exc

    if not records:
        raise DatasetValidationError(f"{path} contains no {record_label} records")
    return records


def _find_duplicate_ids(records: list[Document] | list[RetrievalQuery]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        if record.id in seen:
            duplicates.add(record.id)
        seen.add(record.id)
    return duplicates


def load_corpus(path: str | Path) -> list[Document]:
    """Load documents and reject duplicate IDs."""
    corpus_path = Path(path)
    documents = _load_jsonl(corpus_path, Document, "document")
    duplicates = _find_duplicate_ids(documents)
    if duplicates:
        raise DatasetValidationError(
            f"duplicate document IDs in {corpus_path}: {', '.join(sorted(duplicates))}"
        )
    return documents


def load_queries(path: str | Path) -> list[RetrievalQuery]:
    """Load labeled queries and reject duplicate IDs."""
    queries_path = Path(path)
    queries = _load_jsonl(queries_path, RetrievalQuery, "query")
    duplicates = _find_duplicate_ids(queries)
    if duplicates:
        raise DatasetValidationError(
            f"duplicate query IDs in {queries_path}: {', '.join(sorted(duplicates))}"
        )
    return queries


def load_dataset(
    corpus_path: str | Path, query_path: str | Path
) -> tuple[list[Document], list[RetrievalQuery]]:
    """Load both JSONL files and validate relevance references."""
    documents = load_corpus(corpus_path)
    queries = load_queries(query_path)
    document_ids = {document.id for document in documents}
    missing_references = {
        relevant_id
        for query in queries
        for relevant_id in query.relevant_doc_ids
        if relevant_id not in document_ids
    }
    if missing_references:
        raise DatasetValidationError(
            "queries reference nonexistent document IDs: "
            + ", ".join(sorted(missing_references))
        )
    return documents, queries

