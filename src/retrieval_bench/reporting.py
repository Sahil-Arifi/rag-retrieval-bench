"""JSON, CSV, Markdown, and chart reporting for benchmark results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from retrieval_bench.models import BenchmarkResults, ExperimentResult


class ReportingError(ValueError):
    """Raised when results cannot be ranked or serialized safely."""


def _metric_value(result: ExperimentResult, primary_metric: str) -> float:
    try:
        value = result.metrics[primary_metric]
    except KeyError as exc:
        raise ReportingError(f"result is missing primary metric {primary_metric!r}") from exc
    if not math.isfinite(value):
        raise ReportingError(f"primary metric {primary_metric!r} must be finite")
    return value


def sort_results(
    results: list[ExperimentResult], primary_metric: str
) -> list[ExperimentResult]:
    """Sort quality descending, then prefer lower latency, scale, and chunk settings."""
    if not results:
        raise ReportingError("cannot rank an empty result set")
    metric = primary_metric.lower()
    return sorted(
        results,
        key=lambda result: (
            -_metric_value(result, metric),
            result.mean_query_latency_ms,
            result.number_of_chunks,
            result.chunk_size,
            result.chunk_overlap,
        ),
    )


def select_best(results: list[ExperimentResult], primary_metric: str) -> ExperimentResult:
    return sort_results(results, primary_metric)[0]


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def _markdown_report(results: BenchmarkResults) -> str:
    ranked = sort_results(results.results, results.primary_metric)
    best = ranked[0]
    recall_metrics = sorted(
        (name for name in best.metrics if name.startswith("recall@")),
        key=lambda name: int(name.split("@", maxsplit=1)[1]),
    )
    metric_columns = [*recall_metrics, "mrr@10", "ndcg@10"]
    header = [
        "Rank",
        "Chunk size",
        "Overlap",
        "Chunks",
        *metric_columns,
        "Mean latency (ms)",
        "p95 latency (ms)",
        "Build time (s)",
    ]
    rows: list[list[str]] = []
    for rank, result in enumerate(ranked, start=1):
        rows.append(
            [
                str(rank),
                str(result.chunk_size),
                str(result.chunk_overlap),
                str(result.number_of_chunks),
                *(f"{result.metrics[name]:.4f}" for name in metric_columns),
                f"{result.mean_query_latency_ms:.3f}",
                f"{result.p95_query_latency_ms:.3f}",
                f"{result.embedding_index_build_time_seconds:.3f}",
            ]
        )
    table = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    runtime_lines = [f"- **{name}:** {value}" for name, value in results.runtime.items()]
    return "\n".join(
        [
            "# Retrieval benchmark report",
            "",
            "> **Demonstration dataset only.** These measurements verify the evaluation pipeline; "
            "they are not a statistically meaningful or general model benchmark.",
            "",
            f"Generated: `{results.generated_at.isoformat()}`",
            "",
            "## Best measured configuration",
            "",
            f"The highest-ranked run by `{results.primary_metric}` used chunk size "
            f"**{best.chunk_size}** with overlap **{best.chunk_overlap}**. It measured "
            f"`{results.primary_metric}={best.metrics[results.primary_metric]:.4f}`, mean query "
            f"latency `{best.mean_query_latency_ms:.3f} ms`, and p95 query latency "
            f"`{best.p95_query_latency_ms:.3f} ms` on the runtime below.",
            "",
            "## All experiment configurations",
            "",
            *table,
            "",
            "Construction time includes document embedding and index construction. Query latency "
            "includes single-query encoding, exact chunk search, and source-document deduplication. "
            "Model loading, warmup, chunking, and report generation are excluded.",
            "",
            "## Runtime",
            "",
            *runtime_lines,
            "",
        ]
    )


def _write_plot(path: Path, results: BenchmarkResults) -> None:
    ranked = sort_results(results.results, results.primary_metric)
    recall_metric = "recall@10"
    fig, axis = plt.subplots(figsize=(9, 6))
    for result in ranked:
        axis.scatter(
            result.mean_query_latency_ms,
            result.metrics[recall_metric],
            s=55,
            alpha=0.85,
        )
        axis.annotate(
            f"{result.chunk_size}/{result.chunk_overlap}",
            (result.mean_query_latency_ms, result.metrics[recall_metric]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_title("Recall@10 versus mean query latency")
    axis.set_xlabel("Mean query latency (ms; lower is better)")
    axis.set_ylabel("Recall@10 (higher is better)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".png", dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        fig.savefig(temporary_path, format="png", dpi=160)
        temporary_path.replace(path)
    finally:
        plt.close(fig)
        temporary_path.unlink(missing_ok=True)


def write_reports(results: BenchmarkResults, output_directory: str | Path) -> dict[str, Path]:
    """Write the four required artifacts and return their paths."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_path / "results.json",
        "csv": output_path / "results.csv",
        "markdown": output_path / "report.md",
        "plot": output_path / "recall_vs_latency.png",
    }

    json_payload = json.dumps(
        results.model_dump(mode="json"), indent=2, ensure_ascii=False, allow_nan=False
    )
    _atomic_write_text(paths["json"], json_payload + "\n")

    ranked = sort_results(results.results, results.primary_metric)
    frame = pd.DataFrame([result.flattened() for result in ranked])
    with NamedTemporaryFile("w", newline="", dir=output_path, delete=False) as handle:
        temporary_csv = Path(handle.name)
    try:
        frame.to_csv(temporary_csv, index=False, float_format="%.6f")
        temporary_csv.replace(paths["csv"])
    finally:
        temporary_csv.unlink(missing_ok=True)

    _atomic_write_text(paths["markdown"], _markdown_report(results))
    _write_plot(paths["plot"], results)
    return paths

