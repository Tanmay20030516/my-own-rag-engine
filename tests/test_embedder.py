"""
Tests for rag/embedder.py.

ollama.Client.embed / the module-level ollama.embed are mocked out so
these run without a live Ollama server -- they check the request shape
(prefixes, batching) and response handling, not the model itself.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.embedder import OllamaEmbedder


class _FakeResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


def test_embed_documents_prefixes_and_batches_in_one_call():
    embedder = OllamaEmbedder(model="nomic-embed-text")
    embedder._client = MagicMock()
    embedder._client.embed.return_value = _FakeResponse([[1.0, 2.0], [3.0, 4.0]])

    result = embedder.embed_documents(["hello", "world"])

    embedder._client.embed.assert_called_once_with(
        model="nomic-embed-text",
        input=["search_document: hello", "search_document: world"],
    )
    np.testing.assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])


def test_embed_query_uses_query_prefix_and_returns_single_vector():
    embedder = OllamaEmbedder(model="nomic-embed-text")
    embedder._client = MagicMock()
    embedder._client.embed.return_value = _FakeResponse([[5.0, 6.0]])

    result = embedder.embed_query("what is a vector database?")

    embedder._client.embed.assert_called_once_with(
        model="nomic-embed-text",
        input=["search_query: what is a vector database?"],
    )
    assert result.shape == (2,)
    np.testing.assert_array_equal(result, [5.0, 6.0])


def test_embed_documents_empty_list_skips_the_call():
    embedder = OllamaEmbedder(model="nomic-embed-text")
    embedder._client = MagicMock()

    result = embedder.embed_documents([])

    embedder._client.embed.assert_not_called()
    assert result.shape == (0, 0)


def test_host_uses_a_dedicated_client():
    with patch("rag.embedder.ollama.Client") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        embedder = OllamaEmbedder(model="nomic-embed-text", host="http://example:11434")

    mock_client_cls.assert_called_once_with(host="http://example:11434")
    assert embedder._client is mock_client_cls.return_value


def test_no_host_uses_module_level_client():
    import ollama

    embedder = OllamaEmbedder(model="nomic-embed-text")
    assert embedder._client is ollama
