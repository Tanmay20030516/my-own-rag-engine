"""
Tests for rag/chunker.py.

chunk_fixed and chunk_sentences are pure functions, tested directly.
chunk_semantic needs an embedder; a fake one maps sentences to hand-picked
vectors so the expected breakpoint is known ahead of time, with no
dependency on a running Ollama server.
"""

import numpy as np
import pytest

from rag.chunker import chunk_fixed, chunk_semantic, chunk_sentences


def test_chunk_fixed_splits_into_windows_with_overlap():
    text = " ".join(f"w{i}" for i in range(10))
    chunks = chunk_fixed(text, chunk_size=4, overlap=1)

    assert chunks == [
        "w0 w1 w2 w3",
        "w3 w4 w5 w6",
        "w6 w7 w8 w9",
    ]


def test_chunk_fixed_single_chunk_when_shorter_than_chunk_size():
    text = "one two three"
    assert chunk_fixed(text, chunk_size=10, overlap=2) == ["one two three"]


def test_chunk_fixed_empty_text():
    assert chunk_fixed("", chunk_size=10, overlap=2) == []


def test_chunk_fixed_rejects_overlap_ge_chunk_size():
    with pytest.raises(ValueError):
        chunk_fixed("a b c", chunk_size=4, overlap=4)


def test_chunk_sentences_groups_into_windows():
    text = "One. Two. Three. Four. Five."
    chunks = chunk_sentences(text, max_sentences=2)

    assert chunks == ["One. Two.", "Three. Four.", "Five."]


def test_chunk_sentences_handles_multiple_terminators():
    text = "Is this real? Yes! It is."
    assert chunk_sentences(text, max_sentences=5) == ["Is this real? Yes! It is."]


class _FakeEmbedder:
    """Maps each sentence to a hand-picked vector so the breakpoint is known."""

    def __init__(self, vectors_by_sentence: dict[str, list[float]]):
        self._vectors = vectors_by_sentence

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vectors[t] for t in texts], dtype=np.float64)


def test_chunk_semantic_splits_at_the_outlier_gap():
    # Two clusters of near-identical vectors, with one big jump between
    # sentence 2 and 3 -- that gap should be the single breakpoint.
    sentences = ["A.", "A2.", "A3.", "B.", "B2."]
    vectors = {
        "A.": [1.0, 0.0],
        "A2.": [0.99, 0.01],
        "A3.": [0.98, 0.02],
        "B.": [0.0, 1.0],
        "B2.": [0.01, 0.99],
    }
    embedder = _FakeEmbedder(vectors)
    text = " ".join(sentences)

    chunks = chunk_semantic(text, embedder, breakpoint_percentile=95.0)

    assert chunks == ["A. A2. A3.", "B. B2."]


def test_chunk_semantic_single_sentence_returned_as_is():
    embedder = _FakeEmbedder({})
    assert chunk_semantic("Only one sentence here.", embedder) == ["Only one sentence here."]


def test_chunk_semantic_empty_text():
    embedder = _FakeEmbedder({})
    assert chunk_semantic("", embedder) == []
