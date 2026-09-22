"""A transparent document-level Okapi BM25 baseline for the same relevance labels."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from time import perf_counter
from typing import Any

from retrieval_bench.evaluator import _evaluate_quality
from retrieval_bench.models import Document, QueryResult, RetrievalQuery
from retrieval_bench.provenance import build_provenance


def tokenize(text: str) -> list[str]:
    """Unicode word tokens, casefolded; no stemming or language-specific stopwords."""
    return re.findall(r"\w+", text.casefold())


class BM25Index:
    def __init__(self, documents: list[Document], *, k1: float = 1.2, b: float = 0.75):
        if not documents or not math.isfinite(k1) or k1 <= 0:
            raise ValueError("documents must be nonempty and k1 must be finite and positive")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be finite and between zero and one")
        self.documents = documents
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for position, document in enumerate(documents):
            terms = tokenize(document.text)
            self.lengths.append(len(terms))
            for term, count in Counter(terms).items():
                self.postings[term].append((position, count))
        self.average_length = sum(self.lengths) / len(documents)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if k < 1:
            raise ValueError("k must be positive")
        scores = [0.0] * len(self.documents)
        for term in sorted(set(tokenize(query))):
            postings = self.postings.get(term, [])
            idf = math.log1p((len(self.documents) - len(postings) + 0.5) / (len(postings) + 0.5))
            for position, frequency in postings:
                normalization = 1 - self.b + self.b * self.lengths[position] / self.average_length
                scores[position] += (
                    idf * frequency * (self.k1 + 1) / (frequency + self.k1 * normalization)
                )
        # Zero scores are retained; ties use source order, matching the dense harness.
        order = sorted(range(len(scores)), key=lambda position: (-scores[position], position))
        return [(self.documents[i].id, scores[i]) for i in order[:k]]


def evaluate_bm25(
    documents: list[Document],
    queries: list[RetrievalQuery],
    *,
    k_values: list[int],
    k1: float = 1.2,
    b: float = 0.75,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, Any]:
    if not queries:
        raise ValueError("queries cannot be empty")
    if not k_values or any(k < 1 for k in k_values):
        raise ValueError("k_values must contain positive cutoffs")
    index = BM25Index(documents, k1=k1, b=b)
    results: list[QueryResult] = []
    for query in queries:
        started = clock()
        hits = index.search(query.query, max(10, max(k_values)))
        latency_ms = (clock() - started) * 1000
        ranking = [doc_id for doc_id, _ in hits]
        results.append(
            QueryResult(
                query_id=query.id,
                relevant_doc_ids=query.relevant_doc_ids,
                ranked_doc_ids=ranking,
                scores=[score for _, score in hits],
                latency_ms=latency_ms,
                metrics=_evaluate_quality([ranking], [query.relevant_doc_ids], k_values),
            )
        )
    return {
        "schema_version": 1,
        "method": "Okapi BM25; casefolded Unicode word tokens; text field only",
        "parameters": {"k1": k1, "b": b},
        "provenance": build_provenance(documents, queries, model_name="bm25", model_revision=None),
        "metrics": _evaluate_quality(
            [result.ranked_doc_ids for result in results],
            [query.relevant_doc_ids for query in queries],
            k_values,
        ),
        "query_results": [result.model_dump() for result in results],
    }
