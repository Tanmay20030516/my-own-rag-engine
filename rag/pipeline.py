"""
RAGPipeline — chunk documents, embed and index them, then answer questions
by retrieving the closest chunks and asking a local Ollama LLM to answer
from that context. Every call in this pipeline (embedding + generation)
stays on the local Ollama server -- no external API calls.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import ollama

from rag.chunker import chunk_fixed
from rag.embedder import OllamaEmbedder
from vectordb.index.base import VectorIndex

DEFAULT_SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "If the context doesn't contain the answer, say you don't know."
)


@dataclass(frozen=True)
class RetrievedChunk:
    """One search hit: the chunk's index id, its text, and its distance to the query (lower = closer)."""

    id: int
    text: str
    distance: float


class RAGPipeline:

    def __init__(
        self,
        index: VectorIndex,
        embedder: OllamaEmbedder | None = None,
        chunker: Callable[[str], list[str]] = chunk_fixed,
        llm_model: str = "qwen3:8b",
    ) -> None:
        self.index = index
        self.embedder = embedder or OllamaEmbedder()
        self.chunker = chunker
        self.llm_model = llm_model
        self._chunks: dict[int, str] = {}
        self._next_id = 0
        self._trained = False

    def index_documents(self, docs: list[str]) -> None:
        """Chunk each doc, embed all chunks in one batch, and add them to the index."""
        chunks = [chunk for doc in docs for chunk in self.chunker(doc)]
        if not chunks:
            return

        embeddings = self.embedder.embed_documents(chunks)
        # IVF/PQ need fitted centroids before add(); Flat/HNSW treat this as a
        # no-op. Only the first batch trains -- re-fitting later would invalidate
        # the codes/lists already assigned to stored vectors.
        if not self._trained:
            self.index.train(embeddings)
            self._trained = True

        ids = np.arange(self._next_id, self._next_id + len(chunks))
        self.index.add(embeddings, ids)

        for cid, text in zip(ids, chunks):
            self._chunks[int(cid)] = text
        self._next_id += len(chunks)

    def retrieve(self, question: str, k: int = 5) -> list[RetrievedChunk]:
        """Embed the question and return the k closest chunks, nearest first."""
        if not self._chunks:
            return []

        q_embedding = self.embedder.embed_query(question)
        distances, ids = self.index.search(q_embedding, k)
        return [
            RetrievedChunk(id=int(i), text=self._chunks[int(i)], distance=float(d))
            for d, i in zip(distances, ids)
            if int(i) in self._chunks
        ]

    def query(self, question: str, k: int = 5, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        """Embed the question, retrieve the k closest chunks, and ask the LLM to answer from them."""
        context = "\n\n".join(hit.text for hit in self.retrieve(question, k))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]

        response = ollama.chat(model=self.llm_model, messages=messages)
        return response["message"]["content"]
