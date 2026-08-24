"""Tests for document-level retrieval metrics."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from retrieval_bench.metrics import (
    dcg_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_deduplicates_documents_before_applying_cutoff() -> None:
    ranking = ["doc_a", "doc_a", "doc_b", "doc_c"]

    assert recall_at_k(ranking, {"doc_b", "doc_d"}, 2) == pytest.approx(0.5)


def test_recall_uses_all_relevant_documents_as_denominator() -> None:
    assert recall_at_k(["doc_a", "doc_b"], {"doc_a", "doc_b", "doc_c"}, 2) == pytest.approx(
        2 / 3
    )


def test_reciprocal_rank_uses_unique_document_ranks() -> None:
    ranking = ["doc_a", "doc_a", "doc_b", "doc_c"]

    assert reciprocal_rank(ranking, {"doc_b"}) == pytest.approx(0.5)
    assert reciprocal_rank(ranking, {"doc_b"}, 1) == 0.0


def test_mrr_at_k_applies_cutoff_per_query() -> None:
    rankings = [["doc_a", "doc_b"], ["doc_w", "doc_x", "doc_y", "doc_z"]]
    relevance = [{"doc_a"}, {"doc_z"}]

    assert mrr_at_k(rankings, relevance, 3) == pytest.approx(0.5)
    assert mrr_at_k(rankings, relevance, 10) == pytest.approx(0.625)


def test_dcg_at_k_uses_linear_gain_and_logarithmic_discount() -> None:
    expected = 1.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4)

    assert dcg_at_k([1.0, 1.0, 1.0, 1.0], 3) == pytest.approx(expected)


def test_ndcg_at_k_supports_multiple_relevant_documents() -> None:
    ranking = ["doc_a", "doc_b", "doc_c"]

    assert ndcg_at_k(ranking, {"doc_b", "doc_c"}, 3) == pytest.approx(0.6934264036172708)


def test_ndcg_deduplicates_documents_before_applying_cutoff() -> None:
    ranking = ["doc_a", "doc_a", "doc_b", "doc_c"]
    observed = 1.0 / math.log2(3)
    ideal = 1.0 + 1.0 / math.log2(3)

    assert ndcg_at_k(ranking, {"doc_b", "doc_c"}, 2) == pytest.approx(observed / ideal)


def test_perfect_ndcg_is_one() -> None:
    assert ndcg_at_k(["doc_b", "doc_c", "doc_a"], {"doc_b", "doc_c"}, 3) == 1.0


@pytest.mark.parametrize(
    "metric_call",
    [
        lambda: recall_at_k(["doc_a"], {"doc_a"}, 0),
        lambda: reciprocal_rank(["doc_a"], {"doc_a"}, 0),
        lambda: mrr_at_k([["doc_a"]], [{"doc_a"}], 0),
        lambda: dcg_at_k([1.0], 0),
        lambda: ndcg_at_k(["doc_a"], {"doc_a"}, 0),
    ],
)
def test_metrics_reject_nonpositive_cutoffs(metric_call: Callable[[], float]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        metric_call()


def test_metrics_return_zero_when_no_document_is_relevant() -> None:
    assert recall_at_k(["doc_a"], set(), 1) == 0.0
    assert reciprocal_rank(["doc_a"], set(), 1) == 0.0
    assert ndcg_at_k(["doc_a"], set(), 1) == 0.0


def test_mrr_requires_aligned_nonempty_query_collections() -> None:
    with pytest.raises(ValueError, match="same length"):
        mrr_at_k([["doc_a"]], [], 10)
    with pytest.raises(ValueError, match="at least one query"):
        mrr_at_k([], [], 10)


@pytest.mark.parametrize("scores", [[-1.0], [math.inf], [math.nan]])
def test_dcg_rejects_invalid_relevance_scores(scores: list[float]) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        dcg_at_k(scores, 10)
