"""Tests for ranking and the complete reporting artifact set."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from retrieval_bench.models import BenchmarkResults, ExperimentResult
from retrieval_bench.reporting import ReportingError, select_best, write_reports


def make_result(
    *,
    chunk_size: int,
    overlap: int,
    mrr: float,
    latency: float,
    number_of_chunks: int = 12,
) -> ExperimentResult:
    return ExperimentResult(
        model_name="fake",
        index_backend="numpy",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        number_of_chunks=number_of_chunks,
        embedding_index_build_time_seconds=0.125,
        mean_query_latency_ms=latency,
        p95_query_latency_ms=latency + 1.0,
        query_count=2,
        metrics={
            "recall@1": 0.5,
            "recall@3": 1.0,
            "recall@5": 1.0,
            "recall@10": 1.0,
            "mrr@10": mrr,
            "ndcg@10": 0.75,
        },
    )


def test_select_best_uses_primary_metric_then_latency() -> None:
    slower = make_result(chunk_size=128, overlap=0, mrr=0.8, latency=3.0)
    faster = make_result(chunk_size=256, overlap=32, mrr=0.8, latency=2.0)
    lower_quality = make_result(chunk_size=512, overlap=64, mrr=0.7, latency=1.0)

    assert select_best([slower, lower_quality, faster], "mrr@10") == faster


def test_write_reports_creates_valid_complete_artifacts(tmp_path: Path) -> None:
    first = make_result(chunk_size=128, overlap=0, mrr=0.6, latency=2.5)
    best = make_result(chunk_size=256, overlap=32, mrr=0.9, latency=3.5)
    results = BenchmarkResults(
        generated_at=datetime(2026, 8, 23, tzinfo=UTC),
        primary_metric="mrr@10",
        corpus_path="data/sample/corpus.jsonl",
        queries_path="data/sample/queries.jsonl",
        configuration={"evaluation": {"primary_metric": "mrr@10"}},
        runtime={"python": "test", "platform": "test"},
        results=[first, best],
    )

    paths = write_reports(results, tmp_path)

    assert set(paths) == {"json", "csv", "markdown", "plot"}
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert len(payload["results"]) == 2
    frame = pd.read_csv(paths["csv"])
    assert frame.iloc[0]["chunk_size"] == 256
    report = paths["markdown"].read_text(encoding="utf-8")
    assert "Demonstration dataset only" in report
    assert "chunk size **256** with overlap **32**" in report


def test_reporting_rejects_missing_or_nonfinite_primary_metric() -> None:
    result = make_result(chunk_size=128, overlap=0, mrr=0.5, latency=1.0)
    with pytest.raises(ReportingError, match="missing primary metric"):
        select_best([result], "unknown@10")

    invalid = result.model_copy(update={"metrics": {**result.metrics, "mrr@10": float("nan")}})
    with pytest.raises(ReportingError, match="must be finite"):
        select_best([invalid], "mrr@10")
