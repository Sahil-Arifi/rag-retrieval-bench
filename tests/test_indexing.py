"""Contract tests for the exact vector-index implementations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from retrieval_bench.indexing import FaissIndex, IndexError, NumpyExactIndex, VectorIndex

FloatMatrix = NDArray[np.float32]
IndexFactory = Callable[[], VectorIndex]


def _faiss_index() -> FaissIndex:
    pytest.importorskip("faiss")
    return FaissIndex()


@pytest.fixture(params=[NumpyExactIndex, _faiss_index], ids=["numpy", "faiss"])
def index_factory(request: pytest.FixtureRequest) -> IndexFactory:
    return request.param


def test_numpy_exact_index_returns_expected_cosine_ranking() -> None:
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    queries = np.asarray([[1.0, 0.2, 0.05], [0.1, 0.4, 1.0]], dtype=np.float32)
    index = NumpyExactIndex()
    index.build(vectors)

    scores, indices = index.search(queries, k=4)

    np.testing.assert_array_equal(indices[0], [0, 3, 1, 2])
    np.testing.assert_array_equal(indices[1], [2, 3, 1, 0])
    assert np.all(np.diff(scores, axis=1) <= 0.0)


def test_faiss_and_numpy_return_equivalent_exact_rankings() -> None:
    pytest.importorskip("faiss")
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    queries = np.asarray([[1.0, 0.2, 0.05], [0.1, 0.4, 1.0]], dtype=np.float32)
    numpy_index = NumpyExactIndex()
    faiss_index = FaissIndex()
    numpy_index.build(vectors)
    faiss_index.build(vectors)

    numpy_scores, numpy_indices = numpy_index.search(queries, k=4)
    faiss_scores, faiss_indices = faiss_index.search(queries, k=4)

    np.testing.assert_array_equal(faiss_indices, numpy_indices)
    np.testing.assert_allclose(faiss_scores, numpy_scores, rtol=1e-6, atol=1e-7)


def test_index_uses_cosine_similarity_instead_of_vector_magnitude(
    index_factory: IndexFactory,
) -> None:
    index = index_factory()
    index.build(np.asarray([[2.0, 0.0], [100.0, 100.0]], dtype=np.float32))

    scores, indices = index.search(np.asarray([[3.0, 0.0]], dtype=np.float32), k=2)

    np.testing.assert_array_equal(indices, [[0, 1]])
    np.testing.assert_allclose(scores, [[1.0, 1.0 / np.sqrt(2.0)]], rtol=1e-6, atol=1e-7)


def test_search_clamps_k_to_index_size(index_factory: IndexFactory) -> None:
    index = index_factory()
    index.build(np.eye(3, dtype=np.float32))

    scores, indices = index.search(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), k=10)

    assert scores.shape == (1, 3)
    assert indices.shape == (1, 3)
    assert np.all(indices >= 0)


def test_unique_similarities_are_returned_in_canonical_descending_order(
    index_factory: IndexFactory,
) -> None:
    index = index_factory()
    index.build(
        np.asarray(
            [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
    )

    scores, indices = index.search(np.asarray([[5.0, 0.0]], dtype=np.float32), k=4)

    np.testing.assert_array_equal(indices, [[0, 1, 2, 3]])
    np.testing.assert_allclose(scores, [[1.0, 0.8, 0.0, -1.0]], rtol=1e-6, atol=1e-7)


def test_equal_similarities_use_source_row_as_tie_breaker(index_factory: IndexFactory) -> None:
    index = index_factory()
    index.build(np.asarray([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    scores, indices = index.search(np.asarray([[1.0, 0.0]], dtype=np.float32), k=3)

    np.testing.assert_array_equal(indices, [[0, 1, 2]])
    np.testing.assert_allclose(scores, [[1.0, 1.0, 0.0]], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "vectors",
    [
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        np.empty((2, 0), dtype=np.float32),
    ],
    ids=["rank-one", "no-rows", "no-columns"],
)
def test_build_rejects_invalid_matrix_shapes(vectors: FloatMatrix) -> None:
    with pytest.raises(IndexError, match=r"rank-2 matrix|cannot be empty"):
        NumpyExactIndex().build(vectors)


def test_build_rejects_zero_norm_vectors(index_factory: IndexFactory) -> None:
    index = index_factory()

    with pytest.raises(IndexError, match="zero-norm"):
        index.build(np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32))


def test_search_rejects_invalid_query_shape(index_factory: IndexFactory) -> None:
    index = index_factory()
    index.build(np.eye(2, dtype=np.float32))

    with pytest.raises(IndexError, match="rank-2 matrix"):
        index.search(np.asarray([1.0, 0.0], dtype=np.float32), k=1)


def test_search_rejects_zero_norm_queries(index_factory: IndexFactory) -> None:
    index = index_factory()
    index.build(np.eye(2, dtype=np.float32))

    with pytest.raises(IndexError, match="zero-norm"):
        index.search(np.asarray([[0.0, 0.0]], dtype=np.float32), k=1)


def test_search_rejects_dimension_mismatch(index_factory: IndexFactory) -> None:
    index = index_factory()
    index.build(np.eye(2, dtype=np.float32))

    with pytest.raises(IndexError, match="dimensions do not match"):
        index.search(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), k=1)


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_nonpositive_k(index_factory: IndexFactory, k: int) -> None:
    index = index_factory()
    index.build(np.eye(2, dtype=np.float32))

    with pytest.raises(IndexError, match="k must be positive"):
        index.search(np.asarray([[1.0, 0.0]], dtype=np.float32), k=k)


@pytest.mark.parametrize("index_factory", [NumpyExactIndex, FaissIndex], ids=["numpy", "faiss"])
def test_search_requires_a_built_index(index_factory: IndexFactory) -> None:
    index = index_factory()

    with pytest.raises(IndexError, match="built before search"):
        index.search(np.asarray([[1.0, 0.0]], dtype=np.float32), k=1)
