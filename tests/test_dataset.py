"""Tests for strict JSONL loading and cross-file dataset validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from retrieval_bench.dataset import (
    DatasetValidationError,
    load_corpus,
    load_dataset,
    load_queries,
)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _document(document_id: str = "doc_001") -> dict[str, Any]:
    return {
        "id": document_id,
        "title": "Synthetic document",
        "text": "Fictional text for a retrieval test.",
    }


def _query(
    query_id: str = "q_001", relevant_doc_ids: list[str] | None = None
) -> dict[str, Any]:
    return {
        "id": query_id,
        "query": "Which fictional document is relevant?",
        "relevant_doc_ids": relevant_doc_ids or ["doc_001"],
    }


def test_load_dataset_accepts_valid_jsonl_and_strips_fields(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    query_path = tmp_path / "queries.jsonl"
    corpus_path.write_text(
        "\n"
        + json.dumps(
            {
                "id": " doc_001 ",
                "title": " Synthetic title ",
                "text": " Synthetic body ",
            }
        )
        + "\n\n",
        encoding="utf-8",
    )
    _write_jsonl(
        query_path,
        [
            {
                "id": " q_001 ",
                "query": " Synthetic question? ",
                "relevant_doc_ids": [" doc_001 "],
            }
        ],
    )

    documents, queries = load_dataset(corpus_path, query_path)

    assert [(document.id, document.title, document.text) for document in documents] == [
        ("doc_001", "Synthetic title", "Synthetic body")
    ]
    assert queries[0].id == "q_001"
    assert queries[0].query == "Synthetic question?"
    assert queries[0].relevant_doc_ids == ["doc_001"]


def test_load_corpus_rejects_unknown_fields(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    record = _document()
    record["unexpected"] = "strict models must reject this"
    _write_jsonl(corpus_path, [record])

    with pytest.raises(DatasetValidationError, match="invalid document"):
        load_corpus(corpus_path)


def test_load_queries_rejects_duplicate_relevance_labels(tmp_path: Path) -> None:
    query_path = tmp_path / "queries.jsonl"
    _write_jsonl(query_path, [_query(relevant_doc_ids=["doc_001", "doc_001"])])

    with pytest.raises(DatasetValidationError, match="invalid query"):
        load_queries(query_path)


def test_load_corpus_reports_malformed_json_line_number(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        json.dumps(_document()) + "\n" + '{"id": "broken"\n',
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match=r":2 contains invalid JSON"):
        load_corpus(corpus_path)


def test_load_corpus_rejects_empty_jsonl(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("\n  \n", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="contains no document records"):
        load_corpus(corpus_path)


def test_load_corpus_rejects_duplicate_document_ids(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus_path, [_document("doc_002"), _document("doc_001"), _document("doc_002")])

    with pytest.raises(DatasetValidationError, match=r"duplicate document IDs.*doc_002"):
        load_corpus(corpus_path)


def test_load_queries_rejects_duplicate_query_ids(tmp_path: Path) -> None:
    query_path = tmp_path / "queries.jsonl"
    _write_jsonl(query_path, [_query("q_001"), _query("q_001")])

    with pytest.raises(DatasetValidationError, match=r"duplicate query IDs.*q_001"):
        load_queries(query_path)


def test_load_dataset_rejects_nonexistent_relevance_references(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    query_path = tmp_path / "queries.jsonl"
    _write_jsonl(corpus_path, [_document("doc_present")])
    _write_jsonl(
        query_path,
        [
            _query("q_001", ["doc_missing_b", "doc_present"]),
            _query("q_002", ["doc_missing_a"]),
        ],
    )

    with pytest.raises(DatasetValidationError) as exc_info:
        load_dataset(corpus_path, query_path)

    message = str(exc_info.value)
    assert "nonexistent document IDs" in message
    assert message.endswith("doc_missing_a, doc_missing_b")
