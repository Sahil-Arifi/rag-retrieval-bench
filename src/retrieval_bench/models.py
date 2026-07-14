"""Core immutable data models used throughout the benchmark."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Base model that rejects misspelled fields and supports stable serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Document(StrictModel):
    """One source document in the retrieval corpus."""

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @field_validator("id", "title", "text")
    @classmethod
    def strip_nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must contain non-whitespace text")
        return value


class RetrievalQuery(StrictModel):
    """A labeled query and the source documents considered relevant to it."""

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_doc_ids: list[str] = Field(min_length=1)

    @field_validator("id", "query")
    @classmethod
    def strip_nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must contain non-whitespace text")
        return value

    @field_validator("relevant_doc_ids")
    @classmethod
    def validate_relevant_ids(cls, value: list[str]) -> list[str]:
        cleaned = [doc_id.strip() for doc_id in value]
        if any(not doc_id for doc_id in cleaned):
            raise ValueError("relevant document IDs cannot be blank")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("relevant document IDs must be unique")
        return cleaned


class Chunk(StrictModel):
    """A token-bounded piece of a source document."""

    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    token_count: int = Field(ge=1)


class RetrievedChunk(StrictModel):
    """A scored chunk returned by a vector index."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    score: float
    rank: int = Field(ge=1)


class ExperimentResult(StrictModel):
    """Configuration, quality, timing, and scale measurements from one run."""

    model_name: str
    index_backend: str
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    number_of_chunks: int = Field(gt=0)
    embedding_index_build_time_seconds: float = Field(ge=0)
    mean_query_latency_ms: float = Field(ge=0)
    p95_query_latency_ms: float = Field(ge=0)
    query_count: int = Field(gt=0)
    metrics: dict[str, float]

    def flattened(self) -> dict[str, Any]:
        """Return a CSV-friendly record with metric names promoted to columns."""
        values = self.model_dump(exclude={"metrics"})
        values.update(self.metrics)
        return values


class BenchmarkResults(StrictModel):
    """Complete benchmark output and minimal reproducibility metadata."""

    generated_at: datetime
    primary_metric: str
    corpus_path: str
    queries_path: str
    configuration: dict[str, Any]
    runtime: dict[str, str]
    results: list[ExperimentResult]
