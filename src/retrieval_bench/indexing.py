"""Exact cosine-similarity indexes backed by FAISS or NumPy."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatMatrix = NDArray[np.float32]
IntMatrix = NDArray[np.int64]


class IndexError(ValueError):
    """Raised when vectors or index operations are invalid."""


class VectorIndex(Protocol):
    @property
    def size(self) -> int: ...

    def build(self, vectors: ArrayLike) -> None: ...

    def search(self, queries: ArrayLike, k: int) -> tuple[FloatMatrix, IntMatrix]: ...


def _normalized_matrix(vectors: ArrayLike, *, label: str) -> FloatMatrix:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise IndexError(f"{label} must be a rank-2 matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise IndexError(f"{label} cannot be empty")
    if not np.isfinite(matrix).all():
        raise IndexError(f"{label} must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise IndexError(f"{label} cannot contain zero-norm rows")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _canonicalize(
    scores: FloatMatrix, indices: IntMatrix
) -> tuple[FloatMatrix, IntMatrix]:
    canonical_scores = np.empty_like(scores)
    canonical_indices = np.empty_like(indices)
    for row in range(scores.shape[0]):
        order = np.lexsort((indices[row], -scores[row]))
        canonical_scores[row] = scores[row, order]
        canonical_indices[row] = indices[row, order]
    return canonical_scores, canonical_indices


class NumpyExactIndex:
    """Portable brute-force inner-product index over normalized vectors."""

    def __init__(self) -> None:
        self._vectors: FloatMatrix | None = None

    @property
    def size(self) -> int:
        return 0 if self._vectors is None else self._vectors.shape[0]

    def build(self, vectors: ArrayLike) -> None:
        self._vectors = _normalized_matrix(vectors, label="index vectors")

    def search(self, queries: ArrayLike, k: int) -> tuple[FloatMatrix, IntMatrix]:
        if self._vectors is None:
            raise IndexError("index must be built before search")
        if k <= 0:
            raise IndexError("k must be positive")
        query_matrix = _normalized_matrix(queries, label="query vectors")
        if query_matrix.shape[1] != self._vectors.shape[1]:
            raise IndexError("query and index dimensions do not match")
        actual_k = min(k, self.size)
        all_scores = query_matrix @ self._vectors.T
        all_indices = np.broadcast_to(
            np.arange(self.size, dtype=np.int64), all_scores.shape
        )
        sorted_scores, sorted_indices = _canonicalize(all_scores, all_indices)
        return sorted_scores[:, :actual_k], sorted_indices[:, :actual_k]


class FaissIndex:
    """FAISS IndexFlatIP adapter with the same exact cosine semantics as NumPy."""

    def __init__(self) -> None:
        self._index: object | None = None
        self._dimension: int | None = None

    @property
    def size(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)  # type: ignore[attr-defined]

    def build(self, vectors: ArrayLike) -> None:
        import faiss

        matrix = _normalized_matrix(vectors, label="index vectors")
        self._dimension = matrix.shape[1]
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(matrix)

    def search(self, queries: ArrayLike, k: int) -> tuple[FloatMatrix, IntMatrix]:
        if self._index is None or self._dimension is None:
            raise IndexError("index must be built before search")
        if k <= 0:
            raise IndexError("k must be positive")
        query_matrix = _normalized_matrix(queries, label="query vectors")
        if query_matrix.shape[1] != self._dimension:
            raise IndexError("query and index dimensions do not match")
        actual_k = min(k, self.size)
        scores, indices = self._index.search(query_matrix, actual_k)  # type: ignore[attr-defined]
        return _canonicalize(
            np.asarray(scores, dtype=np.float32), np.asarray(indices, dtype=np.int64)
        )


def create_index(backend: str) -> VectorIndex:
    """Construct the explicitly requested backend without silent fallback."""
    if backend == "faiss":
        return FaissIndex()
    if backend == "numpy":
        return NumpyExactIndex()
    raise IndexError(f"unsupported index backend: {backend}")

