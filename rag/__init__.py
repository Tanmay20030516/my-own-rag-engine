"""
RAG pipeline: chunking, embedding, and retrieval-augmented generation
over the from-scratch vector indexes in vectordb/, backed by a local
Ollama server for both embeddings and generation.
"""

from rag.chunker import chunk_fixed, chunk_semantic, chunk_sentences
from rag.embedder import OllamaEmbedder
from rag.pipeline import RAGPipeline

__all__ = [
    "chunk_fixed",
    "chunk_sentences",
    "chunk_semantic",
    "OllamaEmbedder",
    "RAGPipeline",
]
