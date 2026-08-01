"""Typer command-line interface for validation and benchmark execution."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from retrieval_bench.config import ConfigurationError, load_config
from retrieval_bench.dataset import DatasetValidationError, load_dataset

app = typer.Typer(
    name="retrieval-bench",
    help="Benchmark document-level RAG retrieval across token chunking configurations.",
    no_args_is_help=True,
)
console = Console()


def _load_and_validate(config_path: Path):  # type: ignore[no-untyped-def]
    config = load_config(config_path)
    documents, queries = load_dataset(config.dataset.corpus, config.dataset.queries)
    return config, documents, queries


@app.command()
def validate(
    config: Annotated[
        Path,
        typer.Option("--config", help="Path to the YAML experiment configuration."),
    ] = Path("configs/default.yaml"),
) -> None:
    """Validate YAML, schemas, IDs, relevance references, and experiment pairs."""
    try:
        benchmark_config, documents, queries = _load_and_validate(config)
    except (ConfigurationError, DatasetValidationError) as exc:
        console.print(f"[bold red]Validation failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("[bold green]Validation passed.[/bold green]")
    console.print(
        f"{len(documents)} documents, {len(queries)} queries, "
        f"{len(benchmark_config.experiments.combinations)} experiment configurations."
    )


@app.command()
def run(
    config: Annotated[
        Path,
        typer.Option("--config", help="Path to the YAML experiment configuration."),
    ] = Path("configs/default.yaml"),
) -> None:
    """Execute the full benchmark matrix and write JSON/CSV/Markdown/PNG artifacts."""
    try:
        benchmark_config, documents, queries = _load_and_validate(config)
    except (ConfigurationError, DatasetValidationError) as exc:
        console.print(f"[bold red]Validation failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    from retrieval_bench.embeddings import SentenceTransformerEmbedder
    from retrieval_bench.evaluator import run_experiments
    from retrieval_bench.reporting import select_best, write_reports

    try:
        with console.status(
            f"Loading {benchmark_config.model.name} (the first run may download model files)..."
        ):
            embedder = SentenceTransformerEmbedder(
                benchmark_config.model.name,
                batch_size=benchmark_config.model.batch_size,
                max_sequence_length=(
                    benchmark_config.model.max_sequence_length
                    or max(benchmark_config.experiments.chunk_sizes)
                ),
            )

        total = len(benchmark_config.experiments.combinations)
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Running experiments", total=total)

            def advance(experiment: int, count: int, result) -> None:  # type: ignore[no-untyped-def]
                del experiment, count
                progress.update(
                    task_id,
                    advance=1,
                    description=(
                        f"chunk={result.chunk_size}, overlap={result.chunk_overlap}, "
                        f"{benchmark_config.evaluation.primary_metric}="
                        f"{result.metrics[benchmark_config.evaluation.primary_metric]:.4f}"
                    ),
                )

            results = run_experiments(
                benchmark_config,
                documents,
                queries,
                embedder=embedder,
                progress_callback=advance,
            )
        paths = write_reports(results, benchmark_config.output.directory)
        best = select_best(results.results, results.primary_metric)
    except Exception as exc:
        console.print(f"[bold red]Benchmark failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("[bold green]Benchmark complete.[/bold green]")
    console.print(
        f"Best: chunk={best.chunk_size}, overlap={best.chunk_overlap}, "
        f"{results.primary_metric}={best.metrics[results.primary_metric]:.4f}"
    )
    console.print("Artifacts:")
    for artifact_path in paths.values():
        console.print(f"  {artifact_path}")


if __name__ == "__main__":
    app()
