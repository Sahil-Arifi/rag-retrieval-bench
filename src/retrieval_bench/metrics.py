"""Document-level information-retrieval metrics.

Retrieval produces ranked chunks, but callers pass the corresponding source
document IDs here. Repeated document IDs are removed in rank order before a
cutoff is applied so that multiple chunks from one document cannot consume
multiple document ranks or relevance gains.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def _stable_unique(doc_ids: Sequence[str]) -> list[str]:
    """Return document IDs once each while preserving their first rank."""
    seen: set[str] = set()
    unique: list[str] = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc_id)
    return unique


def recall_at_k(
    ranked_doc_ids: Sequence[str], relevant_doc_ids: Collection[str], k: int
) -> float:
    """Return the fraction of all relevant documents found in the top ``k``.

    An empty relevance set has no attainable relevant documents and therefore
    returns ``0.0``. Dataset validation normally prevents that case.
    """
    _validate_k(k)
    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0

    retrieved = set(_stable_unique(ranked_doc_ids)[:k])
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(
    ranked_doc_ids: Sequence[str],
    relevant_doc_ids: Collection[str],
    k: int | None = None,
) -> float:
    """Return the reciprocal rank of the first relevant source document.

    When ``k`` is omitted the complete ranking is considered. With a cutoff,
    a first relevant result below that cutoff receives zero credit.
    """
    if k is not None:
        _validate_k(k)

    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0

    ranking = _stable_unique(ranked_doc_ids)
    if k is not None:
        ranking = ranking[:k]
    for rank, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mrr_at_k(
    ranked_doc_ids_by_query: Sequence[Sequence[str]],
    relevant_doc_ids_by_query: Sequence[Collection[str]],
    k: int,
) -> float:
    """Return macro mean reciprocal rank at ``k`` across labeled queries."""
    _validate_k(k)
    if len(ranked_doc_ids_by_query) != len(relevant_doc_ids_by_query):
        raise ValueError("rankings and relevance sets must have the same length")
    if not ranked_doc_ids_by_query:
        raise ValueError("mrr_at_k requires at least one query")

    reciprocal_ranks = (
        reciprocal_rank(ranking, relevant, k)
        for ranking, relevant in zip(
            ranked_doc_ids_by_query, relevant_doc_ids_by_query, strict=True
        )
    )
    return math.fsum(reciprocal_ranks) / len(ranked_doc_ids_by_query)


def dcg_at_k(relevance_scores: Sequence[float], k: int) -> float:
    """Return linear-gain discounted cumulative gain at ``k``.

    Binary retrieval evaluation supplies gains of zero or one. Linear gain is
    kept explicit rather than using the alternative exponential-gain formula.
    """
    _validate_k(k)
    scores = [float(score) for score in relevance_scores[:k]]
    if any(not math.isfinite(score) or score < 0.0 for score in scores):
        raise ValueError("relevance scores must be finite and nonnegative")

    return math.fsum(
        score / math.log2(rank + 1) for rank, score in enumerate(scores, start=1)
    )


def ndcg_at_k(
    ranked_doc_ids: Sequence[str], relevant_doc_ids: Collection[str], k: int
) -> float:
    """Return binary normalized discounted cumulative gain at ``k``."""
    _validate_k(k)
    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0

    ranking = _stable_unique(ranked_doc_ids)[:k]
    observed_gains = [1.0 if doc_id in relevant else 0.0 for doc_id in ranking]
    ideal_gains = [1.0] * min(k, len(relevant))
    ideal_dcg = dcg_at_k(ideal_gains, k)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(observed_gains, k) / ideal_dcg
