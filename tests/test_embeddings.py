"""Model-free tests for the production adapter contract and deterministic fake."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from retrieval_bench.embeddings import (
    EmbeddingError,
    FakeEmbedder,
    SentenceTransformerEmbedder,
)


class StubTokenizer:
    def __init__(self) -> None:
        self.model_max_length = 256

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
        **kwargs: object,
    ) -> list[int]:
        del truncation, kwargs
        length = int(text.removeprefix("tokens:")) if text.startswith("tokens:") else 3
        return list(range(length + (2 if add_special_tokens else 0)))


class StubSentenceTransformer:
    last_instance: StubSentenceTransformer | None = None

    def __init__(self, model_name: str, device: str | None = None) -> None:
        del model_name, device
        self.tokenizer = StubTokenizer()
        self._max_seq_length = 256
        self.auto_model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=512)
        )
        self.called_methods: list[str] = []
        StubSentenceTransformer.last_instance = self

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        self._max_seq_length = value
        self.tokenizer.model_max_length = value

    def __getitem__(self, index: int) -> StubSentenceTransformer:
        if index != 0:
            raise IndexError(index)
        return self

    def _vectors(self, texts: Sequence[str], method: str, **kwargs: object) -> np.ndarray:
        del kwargs
        self.called_methods.append(method)
        return np.asarray([[len(text), 1.0] for text in texts], dtype=np.float64)

    def encode_document(self, texts: Sequence[str], **kwargs: object) -> np.ndarray:
        return self._vectors(texts, "document", **kwargs)

    def encode_query(self, texts: Sequence[str], **kwargs: object) -> np.ndarray:
        return self._vectors(texts, "query", **kwargs)

    def encode(self, texts: Sequence[str], **kwargs: object) -> np.ndarray:
        return self._vectors(texts, "generic", **kwargs)


@pytest.fixture
def stub_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = StubSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_production_adapter_uses_verified_architectural_limit(
    stub_sentence_transformers: None,
) -> None:
    del stub_sentence_transformers
    embedder = SentenceTransformerEmbedder("stub", max_sequence_length=512)

    assert embedder.max_input_tokens == 512
    assert embedder.tokenizer.model_max_length == 512  # type: ignore[attr-defined]

    with pytest.raises(EmbeddingError, match="architectural limit of 512"):
        SentenceTransformerEmbedder("stub", max_sequence_length=513)


def test_production_adapter_refuses_silent_input_truncation(
    stub_sentence_transformers: None,
) -> None:
    del stub_sentence_transformers
    embedder = SentenceTransformerEmbedder("stub", max_sequence_length=128)

    with pytest.raises(EmbeddingError, match="refusing silent truncation"):
        embedder.encode_documents(["tokens:127"])


def test_production_adapter_uses_separate_document_and_query_methods(
    stub_sentence_transformers: None,
) -> None:
    del stub_sentence_transformers
    embedder = SentenceTransformerEmbedder("stub", max_sequence_length=128)

    documents = embedder.encode_documents(["alpha", "beta"])
    queries = embedder.encode_queries(["gamma"])

    assert documents.dtype == np.float32
    assert queries.dtype == np.float32
    assert StubSentenceTransformer.last_instance is not None
    assert StubSentenceTransformer.last_instance.called_methods == ["document", "query"]


def test_fake_embedder_is_deterministic_and_model_free() -> None:
    first = FakeEmbedder(dimension=32)
    second = FakeEmbedder(dimension=32)
    texts = ["alpha beta beta", "gamma delta"]

    np.testing.assert_array_equal(first.encode_documents(texts), second.encode_documents(texts))
    np.testing.assert_array_equal(first.encode_queries(texts), first.encode_documents(texts))
    assert first.encode_documents(texts).shape == (2, 32)


def test_fake_embedder_rejects_invalid_dimension() -> None:
    with pytest.raises(ValueError, match="dimension must be positive"):
        FakeEmbedder(dimension=0)

