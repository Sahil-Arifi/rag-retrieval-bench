"""Command-line validation tests that never construct a production model."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from retrieval_bench.cli import app

runner = CliRunner()


def test_cli_validates_included_sample_dataset() -> None:
    result = runner.invoke(app, ["validate", "--config", "configs/default.yaml"])

    assert result.exit_code == 0, result.output
    assert "Validation passed" in result.output
    assert "12 documents, 25 queries, 9 experiment configurations" in result.output


def test_cli_invalid_data_fails_without_writing_artifacts(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    output_path = tmp_path / "artifacts"
    config_path = tmp_path / "config.yaml"
    corpus_path.write_text(
        json.dumps({"id": "doc_1", "title": "Title", "text": "Body"}) + "\n",
        encoding="utf-8",
    )
    queries_path.write_text(
        json.dumps(
            {
                "id": "q_1",
                "query": "Question?",
                "relevant_doc_ids": ["missing_doc"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "name": "unused-in-validation",
                    "batch_size": 4,
                    "normalize_embeddings": True,
                },
                "dataset": {
                    "corpus": str(corpus_path),
                    "queries": str(queries_path),
                },
                "experiments": {"chunk_sizes": [8], "chunk_overlaps": [0]},
                "index": {"backend": "numpy"},
                "evaluation": {
                    "k_values": [1, 10],
                    "primary_metric": "mrr@10",
                },
                "output": {"directory": str(output_path)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["validate", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "Validation failed" in result.output
    assert "nonexistent document IDs" in result.output
    assert not output_path.exists()


def test_cli_reports_invalid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("model: [unterminated\n", encoding="utf-8")

    result = runner.invoke(app, ["validate", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "invalid YAML" in result.output

