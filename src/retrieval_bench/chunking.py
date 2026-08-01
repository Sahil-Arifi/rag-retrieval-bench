"""Deterministic token-based document chunking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from retrieval_bench.models import Chunk, Document


class Tokenizer(Protocol):
    """The small subset of Hugging Face tokenizer behavior required by the chunker."""

    def encode(
        self, text: str, *, add_special_tokens: bool = False, **kwargs: object
    ) -> list[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str: ...


class ChunkingError(ValueError):
    """Raised when text cannot be converted into nonempty token chunks."""


class TokenChunker:
    """Split documents into overlapping token windows using a model tokenizer."""

    def __init__(self, tokenizer: Tokenizer, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise ChunkingError("chunk_overlap cannot be negative")
        special_token_counter = getattr(tokenizer, "num_special_tokens_to_add", None)
        special_tokens = special_token_counter(pair=False) if callable(special_token_counter) else 0
        content_capacity = chunk_size - special_tokens
        if content_capacity <= 0:
            raise ChunkingError("chunk_size must leave room for model special tokens")
        if chunk_overlap >= content_capacity:
            raise ChunkingError(
                "chunk_overlap must be smaller than the content-token capacity after reserving "
                "model special tokens"
            )
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.special_tokens = special_tokens
        self.content_capacity = content_capacity

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Create stable chunk IDs and preserve source-document identity."""
        # Hugging Face warns when a full source document exceeds the model limit even though the
        # purpose of this call is to split that document before embedding. ``verbose=False``
        # suppresses only that misleading pre-chunk warning.
        token_ids = self.tokenizer.encode(
            document.text, add_special_tokens=False, verbose=False
        )
        if not token_ids:
            raise ChunkingError(f"document {document.id!r} produced no tokens")

        chunks: list[Chunk] = []
        step = self.content_capacity - self.chunk_overlap
        for start in range(0, len(token_ids), step):
            window = token_ids[start : start + self.content_capacity]
            if not window:
                continue
            text = self.tokenizer.decode(
                window,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if text:
                chunk_index = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.id}::chunk_{chunk_index:04d}",
                        doc_id=document.id,
                        chunk_index=chunk_index,
                        text=text,
                        token_count=len(window),
                    )
                )
            if start + self.content_capacity >= len(token_ids):
                break

        if not chunks:
            raise ChunkingError(f"document {document.id!r} produced no nonempty chunks")
        return chunks

    def chunk_documents(self, documents: Sequence[Document]) -> list[Chunk]:
        """Chunk a corpus in input order for deterministic index positions."""
        return [chunk for document in documents for chunk in self.chunk_document(document)]
