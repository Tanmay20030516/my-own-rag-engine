"""
Tests for rag/pipeline.py.

Uses a real FlatIndex (fast, no training needed) with a fake embedder and
a mocked ollama.chat, so the pipeline's wiring -- chunk -> embed -> add,
and embed query -> search -> build context -> chat -- is verified without
a live Ollama server.
"""

from unittest.mock import patch

import numpy as np

from rag.pipeline import RAGPipeline
from vectordb.index import FlatIndex


class _FakeEmbedder:
    """Deterministic, case-insensitive 2D embeddings: word count on axis 0, first-char code on axis 1."""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array([[len(t.split()), ord(t[0].lower())] for t in texts], dtype=np.float64)

    def embed_query(self, text: str) -> np.ndarray:
        return np.array([len(text.split()), ord(text[0].lower())], dtype=np.float64)


def _upper_chunker(text: str) -> list[str]:
    return [text.upper()]


def test_index_documents_chunks_embeds_and_stores_text():
    pipeline = RAGPipeline(
        index=FlatIndex(metric="l2"),
        embedder=_FakeEmbedder(),
        chunker=_upper_chunker,
    )

    pipeline.index_documents(["hello world", "second doc"])

    assert pipeline.index._vectors.shape == (2, 2)
    assert pipeline._chunks == {0: "HELLO WORLD", 1: "SECOND DOC"}


def test_index_documents_skips_add_when_no_chunks():
    pipeline = RAGPipeline(
        index=FlatIndex(metric="l2"), embedder=_FakeEmbedder(), chunker=lambda text: []
    )

    pipeline.index_documents([""])

    assert pipeline.index._vectors is None
    assert pipeline._chunks == {}


def test_query_retrieves_context_and_calls_chat():
    pipeline = RAGPipeline(
        index=FlatIndex(metric="l2"),
        embedder=_FakeEmbedder(),
        chunker=_upper_chunker,
        llm_model="qwen3:8b",
    )
    pipeline.index_documents(["hello world", "second doc"])

    with patch("rag.pipeline.ollama.chat") as mock_chat:
        mock_chat.return_value = {"message": {"content": "the answer"}}
        answer = pipeline.query("hello world", k=1)

    assert answer == "the answer"
    call_kwargs = mock_chat.call_args.kwargs
    assert call_kwargs["model"] == "qwen3:8b"
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "HELLO WORLD" in messages[1]["content"]
    assert "Question: hello world" in messages[1]["content"]
