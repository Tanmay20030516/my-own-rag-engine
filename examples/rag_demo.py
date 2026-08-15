"""
End-to-end RAG demo: index a short document, ask a question about it, and
print the answer from a local Ollama LLM.

Requires `ollama serve` running, plus:
    ollama pull nomic-embed-text
    ollama pull qwen3:8b
"""

from rag.chunker import chunk_sentences
from rag.embedder import OllamaEmbedder
from rag.pipeline import RAGPipeline
from vectordb.index import FlatIndex

DOCUMENT = """
A vector database stores high-dimensional embeddings and answers nearest
neighbor queries. Flat indexes search every stored vector exactly, so
they are slow but always correct. IVF indexes cluster vectors with
k-means and only search the nearest few clusters, trading some recall
for speed. Product quantization compresses each vector into a short code
made of sub-vector centroid ids, cutting memory use dramatically at the
cost of approximate distances. HNSW builds a multi-layer graph and does
greedy search through it, giving good recall and speed but at a higher
memory cost than PQ.
"""

QUESTIONS = [
    "What does an IVF index trade off for speed?",
    "How does product quantization save memory?",
]


def main() -> None:
    pipeline = RAGPipeline(
        index=FlatIndex(metric="cosine"),
        embedder=OllamaEmbedder(model="nomic-embed-text"),
        chunker=chunk_sentences,
        llm_model="qwen3:8b",
    )

    pipeline.index_documents([DOCUMENT])
    print(f"Indexed {len(pipeline._chunks)} chunks.\n")

    for question in QUESTIONS:
        answer = pipeline.query(question, k=3)
        print(f"Q: {question}\nA: {answer}\n")


if __name__ == "__main__":
    main()
