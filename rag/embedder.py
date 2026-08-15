"""
OllamaEmbedder — wraps a local Ollama embedding model.

Defaults to nomic-embed-text, which is instruction-tuned for retrieval:
it expects a "search_document: " prefix on indexed text and a
"search_query: " prefix on queries, so the two live in a shared space
tuned for that asymmetry. Skipping the prefixes measurably hurts recall,
so embed_documents/embed_query bake it in rather than leaving it to the
caller.

Requires `ollama serve` running locally and the model pulled
(`ollama pull nomic-embed-text`).
"""

import numpy as np
import ollama


class OllamaEmbedder:

    def __init__(self, model: str = "nomic-embed-text", host: str | None = None) -> None:
        self.model = model
        self._client = ollama.Client(host=host) if host else ollama

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Batch-embed already-prefixed texts, stacked into a (N, D) array."""
        if not texts:
            return np.empty((0, 0), dtype=np.float64)
        response = self._client.embed(model=self.model, input=texts)
        return np.asarray(response.embeddings, dtype=np.float64)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed chunks for indexing, with the document-side task prefix."""
        return self._embed([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query, with the query-side task prefix. Returns shape (D,)."""
        return self._embed([f"search_query: {text}"])[0]
