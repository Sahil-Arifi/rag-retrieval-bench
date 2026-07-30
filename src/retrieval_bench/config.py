"""YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class ConfigurationError(ValueError):
    """Raised when an experiment configuration cannot be parsed or validated."""


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ModelConfig(ConfigModel):
    name: str = Field(min_length=1)
    batch_size: int = Field(gt=0)
    normalize_embeddings: Literal[True] = True


class DatasetConfig(ConfigModel):
    corpus: Path
    queries: Path


class ExperimentsConfig(ConfigModel):
    chunk_sizes: list[int] = Field(min_length=1)
    chunk_overlaps: list[int] = Field(min_length=1)

    @field_validator("chunk_sizes")
    @classmethod
    def validate_chunk_sizes(cls, values: list[int]) -> list[int]:
        if any(value <= 0 for value in values):
            raise ValueError("chunk sizes must all be greater than zero")
        if len(set(values)) != len(values):
            raise ValueError("chunk sizes must be unique")
        return values

    @field_validator("chunk_overlaps")
    @classmethod
    def validate_chunk_overlaps(cls, values: list[int]) -> list[int]:
        if any(value < 0 for value in values):
            raise ValueError("chunk overlaps cannot be negative")
        if len(set(values)) != len(values):
            raise ValueError("chunk overlaps must be unique")
        return values

    @model_validator(mode="after")
    def validate_matrix(self) -> ExperimentsConfig:
        invalid = [
            (size, overlap)
            for size in self.chunk_sizes
            for overlap in self.chunk_overlaps
            if overlap >= size
        ]
        if invalid:
            rendered = ", ".join(f"size={size}, overlap={overlap}" for size, overlap in invalid)
            raise ValueError(f"chunk overlap must be smaller than chunk size: {rendered}")
        return self

    @property
    def combinations(self) -> list[tuple[int, int]]:
        return [(size, overlap) for size in self.chunk_sizes for overlap in self.chunk_overlaps]


class IndexConfig(ConfigModel):
    backend: Literal["faiss", "numpy"] = "faiss"


class EvaluationConfig(ConfigModel):
    k_values: list[int] = Field(min_length=1)
    primary_metric: str = Field(min_length=1)

    @field_validator("primary_metric")
    @classmethod
    def normalize_primary_metric(cls, value: str) -> str:
        return value.lower()

    @field_validator("k_values")
    @classmethod
    def validate_k_values(cls, values: list[int]) -> list[int]:
        if any(value <= 0 for value in values):
            raise ValueError("k values must all be greater than zero")
        if len(set(values)) != len(values):
            raise ValueError("k values must be unique")
        if 10 not in values:
            raise ValueError("k_values must include 10 so MRR@10 and nDCG@10 can be reported")
        return values

    @model_validator(mode="after")
    def validate_primary_metric(self) -> EvaluationConfig:
        supported = {f"recall@{k}" for k in self.k_values} | {"mrr@10", "ndcg@10"}
        if self.primary_metric.lower() not in supported:
            choices = ", ".join(sorted(supported))
            raise ValueError(f"primary_metric must be one of: {choices}")
        return self


class OutputConfig(ConfigModel):
    directory: Path


class BenchmarkConfig(ConfigModel):
    model: ModelConfig
    dataset: DatasetConfig
    experiments: ExperimentsConfig
    index: IndexConfig = IndexConfig()
    evaluation: EvaluationConfig
    output: OutputConfig


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load a strict benchmark configuration from YAML."""
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw_config = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigurationError(f"could not read configuration {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw_config, dict):
        raise ConfigurationError(f"configuration {config_path} must contain a YAML mapping")

    try:
        return BenchmarkConfig.model_validate(raw_config)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid configuration {config_path}:\n{exc}") from exc
