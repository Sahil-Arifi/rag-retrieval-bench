"""Experiment engine for chunking, indexing, retrieval, quality, and latency."""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter

import numpy as np

from retrieval_bench.chunking import TokenChunker
from retrieval_bench.config import BenchmarkConfig
from retrieval_bench.embeddings import Embedder, SentenceTransformerEmbedder
from retrieval_bench.indexing import VectorIndex, create_index
from retrieval_bench.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from retrieval_bench.models import (
    BenchmarkResults,
    Document,
    ExperimentResult,
    RetrievalQuery,
)
from retrieval_bench.retrieval import Retriever

Clock = Callable[[], float]
IndexFactory = Callable[[], VectorIndex]
ProgressCallback = Callable[[int, int, ExperimentResult], None]


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def runtime_metadata() -> dict[str, str]:
    """Capture enough environment context to interpret hardware-dependent timings."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "sentence-transformers": _package_version("sentence-transformers"),
        "torch": _package_version("torch"),
        "faiss-cpu": _package_version("faiss-cpu"),
        "numpy": _package_version("numpy"),
    }


def _evaluate_quality(
    rankings: list[list[str]],
    relevant_documents: list[list[str]],
    k_values: list[int],
) -> dict[str, float]:
    metrics = {
        f"recall@{k}": float(
            np.mean(
                [
                    recall_at_k(ranking, relevant, k)
                    for ranking, relevant in zip(
                        rankings, relevant_documents, strict=True
                    )
                ]
            )
        )
        for k in k_values
    }
    metrics["mrr@10"] = mrr_at_k(rankings, relevant_documents, 10)
    metrics["ndcg@10"] = float(
        np.mean(
            [
                ndcg_at_k(ranking, relevant, 10)
                for ranking, relevant in zip(rankings, relevant_documents, strict=True)
            ]
        )
    )
    return metrics


def run_experiments(
    config: BenchmarkConfig,
    documents: list[Document],
    queries: list[RetrievalQuery],
    *,
    embedder: Embedder | None = None,
    index_factory: IndexFactory | None = None,
    clock: Clock = perf_counter,
    progress_callback: ProgressCallback | None = None,
) -> BenchmarkResults:
    """Run every validated matrix combination and return structured measurements."""
    if not documents:
        raise ValueError("documents cannot be empty")
    if not queries:
        raise ValueError("queries cannot be empty")

    active_embedder = embedder or SentenceTransformerEmbedder(
        config.model.name,
        batch_size=config.model.batch_size,
        max_sequence_length=(
            config.model.max_sequence_length or max(config.experiments.chunk_sizes)
        ),
    )
    factory = index_factory or (lambda: create_index(config.index.backend))
    active_embedder.warm_up()

    combinations = config.experiments.combinations
    max_rank = max(max(config.evaluation.k_values), 10)
    experiment_results: list[ExperimentResult] = []

    for experiment_number, (chunk_size, chunk_overlap) in enumerate(combinations, start=1):
        chunker = TokenChunker(active_embedder.tokenizer, chunk_size, chunk_overlap)
        chunks = chunker.chunk_documents(documents)

        construction_started = clock()
        document_vectors = active_embedder.encode_documents([chunk.text for chunk in chunks])
        index = factory()
        index.build(document_vectors)
        construction_seconds = clock() - construction_started

        retriever = Retriever(active_embedder, index, chunks)
        rankings: list[list[str]] = []
        relevant_documents: list[list[str]] = []
        query_latencies_ms: list[float] = []

        for query in queries:
            query_started = clock()
            hits = retriever.retrieve(query.query, max_rank)
            query_latencies_ms.append((clock() - query_started) * 1_000)
            rankings.append([hit.doc_id for hit in hits])
            relevant_documents.append(query.relevant_doc_ids)

        result = ExperimentResult(
            model_name=active_embedder.model_name,
            index_backend=config.index.backend,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            number_of_chunks=len(chunks),
            embedding_index_build_time_seconds=float(construction_seconds),
            mean_query_latency_ms=float(np.mean(query_latencies_ms)),
            p95_query_latency_ms=float(
                np.percentile(query_latencies_ms, 95, method="linear")
            ),
            query_count=len(queries),
            metrics=_evaluate_quality(
                rankings, relevant_documents, config.evaluation.k_values
            ),
        )
        experiment_results.append(result)
        if progress_callback is not None:
            progress_callback(experiment_number, len(combinations), result)

    return BenchmarkResults(
        generated_at=datetime.now(UTC),
        primary_metric=config.evaluation.primary_metric.lower(),
        corpus_path=str(config.dataset.corpus),
        queries_path=str(config.dataset.queries),
        configuration=config.model_dump(mode="json"),
        runtime=runtime_metadata(),
        results=experiment_results,
    )
