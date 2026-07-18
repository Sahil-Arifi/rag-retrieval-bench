"""Embedding abstractions for production and deterministic offline tests."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from retrieval_bench.chunking import Tokenizer

FloatMatrix = NDArray[np.float32]


class EmbeddingError(RuntimeError):
    """Raised when text cannot be embedded without violating model constraints."""


class Embedder(Protocol):
    """Shared interface for interchangeable production and test embedders."""

    model_name: str
    tokenizer: Tokenizer

    def encode_documents(self, texts: Sequence[str]) -> FloatMatrix: ...

    def encode_queries(self, texts: Sequence[str]) -> FloatMatrix: ...

    def warm_up(self) -> None: ...


class SentenceTransformerEmbedder:
    """SentenceTransformers adapter with separate document/query code paths."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 64,
        max_sequence_length: int | None = None,
        device: str | None = None,
    ) -> None:
        # Kept local so importing the package or running unit tests never loads a model.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self.tokenizer = self._model.tokenizer

        hard_limits: list[int] = []
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and tokenizer_limit < 1_000_000:
            hard_limits.append(tokenizer_limit)
        try:
            position_limit = self._model[0].auto_model.config.max_position_embeddings
        except (AttributeError, IndexError, TypeError):
            position_limit = None
        if isinstance(position_limit, int):
            hard_limits.append(position_limit)

        architectural_limit = min(hard_limits) if hard_limits else self._model.max_seq_length
        requested_limit = max_sequence_length or self._model.max_seq_length
        if requested_limit > architectural_limit:
            raise EmbeddingError(
                f"requested input budget {requested_limit} exceeds the model's verified "
                f"architectural limit of {architectural_limit} tokens"
            )
        self.max_input_tokens = requested_limit
        self._model.max_seq_length = requested_limit

    def _validate_input_lengths(self, texts: Sequence[str]) -> None:
        for index, text in enumerate(texts):
            token_ids = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
            if len(token_ids) > self.max_input_tokens:
                raise EmbeddingError(
                    f"input {index} has {len(token_ids)} model tokens, exceeding the configured "
                    f"limit of {self.max_input_tokens}; refusing silent truncation"
                )

    def _encode(self, texts: Sequence[str], method_name: str) -> FloatMatrix:
        if not texts:
            raise EmbeddingError("cannot encode an empty text collection")
        self._validate_input_lengths(texts)
        encode_method = getattr(self._model, method_name, self._model.encode)
        vectors = encode_method(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> FloatMatrix:
        return self._encode(texts, "encode_document")

    def encode_queries(self, texts: Sequence[str]) -> FloatMatrix:
        return self._encode(texts, "encode_query")

    def warm_up(self) -> None:
        self.encode_queries(["retrieval benchmark warmup"])


class WhitespaceTokenizer:
    """Minimal reversible tokenizer used by FakeEmbedder and chunking tests."""

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False, **_: object) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        for token in text.split():
            if token not in self._token_to_id:
                token_id = len(self._token_to_id)
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            ids.append(self._token_to_id[token])
        return ids

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(self._id_to_token[token_id] for token_id in token_ids)

    def num_special_tokens_to_add(self, *, pair: bool = False) -> int:
        del pair
        return 0


class FakeEmbedder:
    """Deterministic hashing embedder with no model or network dependency."""

    model_name = "fake-hashing-embedder"

    def __init__(self, dimension: int = 128) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.tokenizer = WhitespaceTokenizer()

    def _encode(self, texts: Sequence[str]) -> FloatMatrix:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            terms = re.findall(r"[a-z0-9]+", text.lower())
            for term in terms:
                digest = hashlib.sha256(term.encode("utf-8")).digest()
                primary = int.from_bytes(digest[:4], "big") % self.dimension
                secondary = int.from_bytes(digest[4:8], "big") % self.dimension
                sign = 1.0 if digest[8] & 1 else -1.0
                vectors[row, primary] += 1.0
                vectors[row, secondary] += 0.25 * sign
            if not np.any(vectors[row]):
                vectors[row, 0] = 1.0
        return vectors

    def encode_documents(self, texts: Sequence[str]) -> FloatMatrix:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> FloatMatrix:
        return self._encode(texts)

    def warm_up(self) -> None:
        return None

