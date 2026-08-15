"""
RAGPipeline — chunk documents, embed and index them, then answer questions
by retrieving the closest chunks and asking a local Ollama LLM to answer
from that context. Every call in this pipeline (embedding + generation)
stays on the local Ollama server -- no external API calls.
"""

from collections.abc import Callable

import numpy as np
import ollama

from rag.chunker import chunk_fixed
from rag.embedder import OllamaEmbedder
from vectordb.index.base import VectorIndex

DEFAULT_SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "If the context doesn't contain the answer, say you don't know."
)


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

    def index_documents(self, docs: list[str]) -> None:
        """Chunk each doc, embed all chunks in one batch, and add them to the index."""
        chunks = [chunk for doc in docs for chunk in self.chunker(doc)]
        if not chunks:
            return

        embeddings = self.embedder.embed_documents(chunks)
        ids = np.arange(self._next_id, self._next_id + len(chunks))
        self.index.add(embeddings, ids)

        for cid, text in zip(ids, chunks):
            self._chunks[int(cid)] = text
        self._next_id += len(chunks)

    def query(self, question: str, k: int = 5, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        """Embed the question, retrieve the k closest chunks, and ask the LLM to answer from them."""
        q_embedding = self.embedder.embed_query(question)
        _, ids = self.index.search(q_embedding, k)

        context = "\n\n".join(self._chunks[int(i)] for i in ids if int(i) in self._chunks)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]

        response = ollama.chat(model=self.llm_model, messages=messages)
        return response["message"]["content"]
