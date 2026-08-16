"""
Tests for RAGPipeline.retrieve() and the train-on-first-ingest behaviour
added for the web UI.

Uses a fake embedder and a spy index so the ordering, id-to-text mapping,
and training rules are checked without a live Ollama server.
"""

import numpy as np

from rag.pipeline import RAGPipeline
from vectordb.index import FlatIndex, IVFIndex


class _FakeEmbedder:
    """Same deterministic 2D scheme as test_pipeline.py: word count, then first-char code."""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array([[len(t.split()), ord(t[0].lower())] for t in texts], dtype=np.float64)

    def embed_query(self, text: str) -> np.ndarray:
        return np.array([len(text.split()), ord(text[0].lower())], dtype=np.float64)


def _one_chunk_per_doc(text: str) -> list[str]:
    return [text]


def _pipeline(index=None) -> RAGPipeline:
    return RAGPipeline(
        index=index or FlatIndex(metric="l2"),
        embedder=_FakeEmbedder(),
        chunker=_one_chunk_per_doc,
    )


def test_retrieve_returns_nearest_chunk_first():
    pipeline = _pipeline()
    pipeline.index_documents(["alpha one", "beta two", "gamma three"])

    hits = pipeline.retrieve("alpha one", k=3)

    assert [h.text for h in hits][0] == "alpha one"
    assert hits[0].distance <= hits[1].distance <= hits[2].distance


def test_retrieve_maps_ids_back_to_their_text():
    pipeline = _pipeline()
    pipeline.index_documents(["alpha one", "beta two"])

    for hit in pipeline.retrieve("beta two", k=2):
        assert pipeline._chunks[hit.id] == hit.text


def test_retrieve_on_empty_pipeline_returns_nothing():
    assert _pipeline().retrieve("anything", k=5) == []


def test_retrieve_caps_k_at_the_number_of_chunks():
    pipeline = _pipeline()
    pipeline.index_documents(["only one"])

    assert len(pipeline.retrieve("only one", k=10)) == 1


def test_index_documents_trains_an_index_that_needs_it():
    # IVF asserts that train() ran before add(); reaching search() proves it did.
    pipeline = _pipeline(IVFIndex(nlist=2, nprobe=2, metric="l2"))
    pipeline.index_documents(["alpha one", "beta two", "gamma three", "delta four"])

    assert pipeline.retrieve("alpha one", k=1)[0].text == "alpha one"


def test_index_documents_trains_only_on_the_first_batch():
    """Re-fitting centroids later would invalidate the assignments already stored."""
    trained_with = []

    class _SpyIndex(FlatIndex):
        def train(self, vectors):
            trained_with.append(vectors.shape[0])

    pipeline = _pipeline(_SpyIndex(metric="l2"))
    pipeline.index_documents(["alpha one", "beta two"])
    pipeline.index_documents(["gamma three"])

    assert trained_with == [2]


def test_second_ingest_keeps_earlier_chunks_retrievable():
    pipeline = _pipeline()
    pipeline.index_documents(["alpha one"])
    pipeline.index_documents(["beta two"])

    assert {h.text for h in pipeline.retrieve("alpha one", k=2)} == {"alpha one", "beta two"}
