"""Integration tests for the deterministic experiment engine."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest

from retrieval_bench.config import BenchmarkConfig
from retrieval_bench.embeddings import FakeEmbedder
from retrieval_bench.evaluator import run_experiments
from retrieval_bench.indexing import NumpyExactIndex
from retrieval_bench.models import Document, ExperimentResult, RetrievalQuery
from retrieval_bench.reporting import select_best


class ScriptedClock:
    """Return caller-supplied timestamps and expose the number consumed."""

    def __init__(self, timestamps: Sequence[float]) -> None:
        self._timestamps = iter(timestamps)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return next(self._timestamps)


class InsertionOrderIndex:
    """Rank chunks by insertion order to make document deduplication observable."""

    def __init__(self) -> None:
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def build(self, vectors: object) -> None:
        matrix = np.asarray(vectors)
        self._size = int(matrix.shape[0])

    def search(self, queries: object, k: int) -> tuple[np.ndarray, np.ndarray]:
        query_count = int(np.asarray(queries).shape[0])
        actual_k = min(k, self._size)
        indices = np.arange(actual_k, dtype=np.int64)
        scores = np.arange(self._size, self._size - actual_k, -1, dtype=np.float32)
        return (
            np.broadcast_to(scores, (query_count, actual_k)).copy(),
            np.broadcast_to(indices, (query_count, actual_k)).copy(),
        )


def _config(
    *,
    chunk_sizes: Sequence[int] = (2, 3),
    chunk_overlaps: Sequence[int] = (0, 1),
) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "model": {
                "name": "unused-by-fake-embedder",
                "batch_size": 8,
                "normalize_embeddings": True,
            },
            "dataset": {
                "corpus": "data/sample/corpus.jsonl",
                "queries": "data/sample/queries.jsonl",
            },
            "experiments": {
                "chunk_sizes": list(chunk_sizes),
                "chunk_overlaps": list(chunk_overlaps),
            },
            "index": {"backend": "numpy"},
            "evaluation": {
                "k_values": [1, 3, 5, 10],
                "primary_metric": "MRR@10",
            },
            "output": {"directory": "artifacts"},
        }
    )


def _separable_documents() -> list[Document]:
    return [
        Document(id="doc_alpha", title="Alpha", text="alpha alpha alpha alpha"),
        Document(id="doc_beta", title="Beta", text="beta beta beta beta"),
        Document(id="doc_gamma", title="Gamma", text="gamma gamma gamma gamma"),
    ]


def _separable_queries() -> list[RetrievalQuery]:
    return [
        RetrievalQuery(id="q_alpha", query="alpha", relevant_doc_ids=["doc_alpha"]),
        RetrievalQuery(id="q_beta", query="beta", relevant_doc_ids=["doc_beta"]),
        RetrievalQuery(id="q_gamma", query="gamma", relevant_doc_ids=["doc_gamma"]),
    ]


def _matrix_clock(experiment_count: int, query_count: int) -> ScriptedClock:
    query_seconds = (1 / 1024, 2 / 1024, 4 / 1024)
    assert query_count == len(query_seconds)

    timestamps: list[float] = []
    cursor = 0.0
    for _ in range(experiment_count):
        timestamps.extend((cursor, cursor + 0.25))
        cursor += 1.0
        for duration in query_seconds:
            timestamps.extend((cursor, cursor + duration))
            cursor += 1.0
    return ScriptedClock(timestamps)


def test_run_experiments_covers_full_matrix_with_deterministic_results() -> None:
    config = _config()
    documents = _separable_documents()
    queries = _separable_queries()
    expected_combinations = [(2, 0), (2, 1), (3, 0), (3, 1)]
    progress: list[tuple[int, int, int, int]] = []

    def record_progress(current: int, total: int, result: ExperimentResult) -> None:
        progress.append((current, total, result.chunk_size, result.chunk_overlap))

    first_clock = _matrix_clock(len(expected_combinations), len(queries))
    first = run_experiments(
        config,
        documents,
        queries,
        embedder=FakeEmbedder(dimension=256),
        index_factory=NumpyExactIndex,
        clock=first_clock,
        progress_callback=record_progress,
    )
    second = run_experiments(
        config,
        documents,
        queries,
        embedder=FakeEmbedder(dimension=256),
        index_factory=NumpyExactIndex,
        clock=_matrix_clock(len(expected_combinations), len(queries)),
    )

    assert [(result.chunk_size, result.chunk_overlap) for result in first.results] == (
        expected_combinations
    )
    assert [result.number_of_chunks for result in first.results] == [6, 9, 6, 6]
    assert progress == [
        (1, 4, 2, 0),
        (2, 4, 2, 1),
        (3, 4, 3, 0),
        (4, 4, 3, 1),
    ]
    assert first_clock.calls == len(expected_combinations) * (2 + 2 * len(queries))
    assert first.results == second.results

    query_latencies_ms = np.asarray([1 / 1024, 2 / 1024, 4 / 1024]) * 1_000
    expected_metrics = {
        "recall@1": 1.0,
        "recall@3": 1.0,
        "recall@5": 1.0,
        "recall@10": 1.0,
        "mrr@10": 1.0,
        "ndcg@10": 1.0,
    }
    for result in first.results:
        assert result.model_name == "fake-hashing-embedder"
        assert result.query_count == len(queries)
        assert result.embedding_index_build_time_seconds == pytest.approx(0.25)
        assert result.mean_query_latency_ms == pytest.approx(np.mean(query_latencies_ms))
        assert result.p95_query_latency_ms == pytest.approx(
            np.percentile(query_latencies_ms, 95, method="linear")
        )
        assert result.metrics == pytest.approx(expected_metrics)

    best = select_best(first.results, first.primary_metric)
    assert (best.chunk_size, best.chunk_overlap) == (2, 0)


def test_evaluator_ranks_unique_source_documents_instead_of_repeated_chunks() -> None:
    config = _config(chunk_sizes=(2,), chunk_overlaps=(1,))
    documents = [
        Document(
            id="doc_noise",
            title="Repeated noise",
            text="noise noise noise noise noise noise",
        ),
        Document(id="doc_relevant", title="Target", text="target"),
        Document(id="doc_other", title="Other", text="other"),
    ]
    queries = [
        RetrievalQuery(
            id="q_target",
            query="target",
            relevant_doc_ids=["doc_relevant"],
        )
    ]
    clock = ScriptedClock((0.0, 0.25, 1.0, 1.0 + 1 / 1024))

    benchmark = run_experiments(
        config,
        documents,
        queries,
        embedder=FakeEmbedder(dimension=64),
        index_factory=InsertionOrderIndex,
        clock=clock,
    )

    result = benchmark.results[0]
    assert result.number_of_chunks == 7
    assert result.metrics["recall@1"] == 0.0
    assert result.metrics["recall@3"] == 1.0
    assert result.metrics["recall@5"] == 1.0
    assert result.metrics["recall@10"] == 1.0
    assert result.metrics["mrr@10"] == pytest.approx(0.5)
    assert result.metrics["ndcg@10"] == pytest.approx(1 / math.log2(3))
