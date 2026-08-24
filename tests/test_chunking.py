"""Tests for deterministic token-window chunking without model downloads."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from retrieval_bench.chunking import ChunkingError, TokenChunker
from retrieval_bench.models import Document


class IntegerTokenizer:
    """A reversible tokenizer for text written as ``t0 t1 ...``."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [int(token.removeprefix("t")) for token in text.split()]

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return " ".join(f"t{token_id}" for token_id in token_ids)


class SpecialTokenTokenizer(IntegerTokenizer):
    def __init__(self, special_token_count: int) -> None:
        self.special_token_count = special_token_count

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        assert pair is False
        return self.special_token_count


class SelectivelyBlankTokenizer(IntegerTokenizer):
    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        if token_ids and token_ids[0] == 0:
            return "   "
        return super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )


class EmptyTokenizer(IntegerTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return []


def _document(document_id: str, token_count: int) -> Document:
    return Document(
        id=document_id,
        title=f"Synthetic {document_id}",
        text=" ".join(f"t{token_id}" for token_id in range(token_count)),
    )


def test_chunk_boundaries_include_short_final_window() -> None:
    chunks = TokenChunker(IntegerTokenizer(), chunk_size=3, chunk_overlap=0).chunk_document(
        _document("doc_a", 7)
    )

    assert [chunk.text for chunk in chunks] == ["t0 t1 t2", "t3 t4 t5", "t6"]
    assert [chunk.token_count for chunk in chunks] == [3, 3, 1]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]


def test_chunk_overlap_repeats_exact_token_windows() -> None:
    chunks = TokenChunker(IntegerTokenizer(), chunk_size=4, chunk_overlap=2).chunk_document(
        _document("doc_a", 8)
    )

    assert [chunk.text for chunk in chunks] == [
        "t0 t1 t2 t3",
        "t2 t3 t4 t5",
        "t4 t5 t6 t7",
    ]
    assert [chunk.token_count for chunk in chunks] == [4, 4, 4]


def test_blank_decoded_windows_never_create_empty_chunks() -> None:
    chunks = TokenChunker(
        SelectivelyBlankTokenizer(), chunk_size=2, chunk_overlap=0
    ).chunk_document(_document("doc_a", 4))

    assert [chunk.text for chunk in chunks] == ["t2 t3"]
    assert [chunk.chunk_id for chunk in chunks] == ["doc_a::chunk_0000"]
    assert all(chunk.text.strip() and chunk.token_count > 0 for chunk in chunks)


def test_tokenizer_that_produces_no_tokens_is_rejected() -> None:
    chunker = TokenChunker(EmptyTokenizer(), chunk_size=4, chunk_overlap=0)

    with pytest.raises(ChunkingError, match="produced no tokens"):
        chunker.chunk_document(_document("doc_empty", 2))


def test_chunk_ids_and_corpus_order_are_deterministic() -> None:
    chunker = TokenChunker(IntegerTokenizer(), chunk_size=2, chunk_overlap=0)
    documents = [_document("doc_b", 3), _document("doc_a", 2)]

    first = chunker.chunk_documents(documents)
    second = chunker.chunk_documents(documents)

    assert first == second
    assert [chunk.chunk_id for chunk in first] == [
        "doc_b::chunk_0000",
        "doc_b::chunk_0001",
        "doc_a::chunk_0000",
    ]
    assert [chunk.doc_id for chunk in first] == ["doc_b", "doc_b", "doc_a"]


def test_special_tokens_reduce_content_capacity() -> None:
    chunker = TokenChunker(
        SpecialTokenTokenizer(special_token_count=2),
        chunk_size=5,
        chunk_overlap=1,
    )

    chunks = chunker.chunk_document(_document("doc_a", 7))

    assert chunker.special_tokens == 2
    assert chunker.content_capacity == 3
    assert [chunk.text for chunk in chunks] == [
        "t0 t1 t2",
        "t2 t3 t4",
        "t4 t5 t6",
    ]
    assert all(chunk.token_count <= 3 for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "error_match"),
    [
        (0, 0, "chunk_size must be greater than zero"),
        (4, -1, "chunk_overlap cannot be negative"),
        (2, 0, "leave room for model special tokens"),
        (5, 3, "smaller than the content-token capacity"),
    ],
)
def test_invalid_sizes_respect_special_token_budget(
    chunk_size: int, chunk_overlap: int, error_match: str
) -> None:
    with pytest.raises(ChunkingError, match=error_match):
        TokenChunker(
            SpecialTokenTokenizer(special_token_count=2),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
