"""Import a user-provided BEIR directory without downloading data or model weights."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from retrieval_bench.models import Document, RetrievalQuery


def import_beir(source: Path, qrels: Path, destination: Path) -> tuple[int, int]:
    """Import positive qrels as binary relevance; preserve all corpus negatives.

    Only queries with positive judgments in the selected split are evaluated.
    A fresh destination prevents accidental replacement of previous inputs.
    """
    documents = []
    for line in (source / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        documents.append(
            Document(id=raw["_id"], title=raw.get("title") or raw["_id"], text=raw["text"])
        )
    raw_queries = {}
    for line in (source / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        if raw["_id"] in raw_queries:
            raise ValueError("duplicate BEIR query ID")
        raw_queries[raw["_id"]] = raw["text"]
    relevant: dict[str, list[str]] = {}
    with qrels.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"query-id", "corpus-id", "score"} <= set(reader.fieldnames or []):
            raise ValueError("qrels needs query-id, corpus-id, score TSV columns")
        for row in reader:
            if any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in ("query-id", "corpus-id", "score")
            ):
                raise ValueError("qrels rows must include a query ID, corpus ID, and numeric score")
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError("qrels scores must be finite")
            if score > 0:
                labels = relevant.setdefault(row["query-id"], [])
                if row["corpus-id"] not in labels:
                    labels.append(row["corpus-id"])
    queries = [
        RetrievalQuery(id=qid, query=raw_queries[qid], relevant_doc_ids=labels)
        for qid, labels in relevant.items()
    ]
    if not documents or not queries:
        raise ValueError("BEIR input must have documents and queries with positive judgments")
    if len({doc.id for doc in documents}) != len(documents):
        raise ValueError("duplicate BEIR corpus ID")
    if len({query.id for query in queries}) != len(queries):
        raise ValueError("duplicate normalized BEIR query ID")
    document_ids = {doc.id for doc in documents}
    if any(set(query.relevant_doc_ids) - document_ids for query in queries):
        raise ValueError("qrels references nonexistent corpus IDs")
    destination.mkdir(parents=True, exist_ok=False)
    for filename, records in (("corpus.jsonl", documents), ("queries.jsonl", queries)):
        (destination / filename).write_text(
            "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
        )
    return len(documents), len(queries)
