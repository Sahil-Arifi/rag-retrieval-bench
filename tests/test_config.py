"""Tests for cross-field experiment configuration validation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from retrieval_bench.config import BenchmarkConfig


def valid_config() -> dict[str, Any]:
    return {
        "model": {
            "name": "fake",
            "batch_size": 8,
            "normalize_embeddings": True,
            "max_sequence_length": 512,
        },
        "dataset": {"corpus": "corpus.jsonl", "queries": "queries.jsonl"},
        "experiments": {"chunk_sizes": [128, 256], "chunk_overlaps": [0, 32]},
        "index": {"backend": "numpy"},
        "evaluation": {"k_values": [1, 3, 10], "primary_metric": "MRR@10"},
        "output": {"directory": "artifacts"},
    }


def test_config_normalizes_primary_metric_and_preserves_matrix_order() -> None:
    config = BenchmarkConfig.model_validate(valid_config())

    assert config.evaluation.primary_metric == "mrr@10"
    assert config.experiments.combinations == [
        (128, 0),
        (128, 32),
        (256, 0),
        (256, 32),
    ]


def test_config_rejects_overlap_not_smaller_than_every_chunk_size() -> None:
    payload = deepcopy(valid_config())
    payload["experiments"]["chunk_overlaps"] = [0, 128]

    with pytest.raises(ValidationError, match="overlap must be smaller than chunk size"):
        BenchmarkConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "values", "message"),
    [
        ("chunk_sizes", [128, 128], "chunk sizes must be unique"),
        ("chunk_sizes", [0, 128], "greater than zero"),
        ("chunk_overlaps", [0, 0], "chunk overlaps must be unique"),
        ("chunk_overlaps", [-1, 0], "cannot be negative"),
    ],
)
def test_config_rejects_invalid_experiment_values(
    field: str, values: list[int], message: str
) -> None:
    payload = deepcopy(valid_config())
    payload["experiments"][field] = values

    with pytest.raises(ValidationError, match=message):
        BenchmarkConfig.model_validate(payload)


def test_config_rejects_model_budget_below_largest_chunk() -> None:
    payload = deepcopy(valid_config())
    payload["model"]["max_sequence_length"] = 128

    with pytest.raises(ValidationError, match="at least the largest configured chunk size"):
        BenchmarkConfig.model_validate(payload)


def test_config_requires_normalized_embeddings_and_mandatory_metrics() -> None:
    payload = deepcopy(valid_config())
    payload["model"]["normalize_embeddings"] = False
    with pytest.raises(ValidationError, match="Input should be True"):
        BenchmarkConfig.model_validate(payload)

    payload = deepcopy(valid_config())
    payload["evaluation"]["k_values"] = [1, 3, 5]
    with pytest.raises(ValidationError, match="must include 10"):
        BenchmarkConfig.model_validate(payload)

