"""Chunk-level vector retrieval collapsed into exact document rankings."""

from __future__ import annotations

from retrieval_bench.embeddings import Embedder
from retrieval_bench.indexing import VectorIndex
from retrieval_bench.models import Chunk, RetrievedChunk


class Retriever:
    """Retrieve chunks, then keep the highest-scoring chunk for each source document."""

    def __init__(self, embedder: Embedder, index: VectorIndex, chunks: list[Chunk]) -> None:
        if index.size != len(chunks):
            raise ValueError("index size must match the chunk metadata count")
        self.embedder = embedder
        self.index = index
        self.chunks = chunks

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the top-k unique documents using exhaustive chunk candidates."""
        if k <= 0:
            raise ValueError("k must be positive")
        query_vector = self.embedder.encode_queries([query])
        scores, indices = self.index.search(query_vector, self.index.size)

        seen_documents: set[str] = set()
        document_hits: list[RetrievedChunk] = []
        for score, index_position in zip(scores[0], indices[0], strict=True):
            chunk = self.chunks[int(index_position)]
            if chunk.doc_id in seen_documents:
                continue
            seen_documents.add(chunk.doc_id)
            document_hits.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=float(score),
                    rank=len(document_hits) + 1,
                )
            )
            if len(document_hits) == k:
                break
        return document_hits

