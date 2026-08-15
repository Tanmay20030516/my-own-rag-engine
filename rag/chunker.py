"""
Text chunking strategies for the RAG pipeline.

chunk_fixed and chunk_sentences split on surface structure (words,
sentence boundaries) and are free -- no embedder needed. chunk_semantic
splits on meaning: it embeds each sentence and cuts wherever consecutive
sentences drift apart in embedding space, so it costs one embedding call
per sentence but tends to keep topically coherent text together instead
of cutting mid-thought at a fixed word count.
"""

import re

import numpy as np

from vectordb.distance import EPS

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split on whitespace following ./!/? -- good enough without a full sentence tokenizer."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip())]
    return [s for s in sentences if s]


def chunk_fixed(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Sliding window over words: chunk_size words per chunk, overlap words shared with the next chunk."""
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")

    words = text.split()
    if not words:
        return []

    stride = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), stride):
        chunk = words[start : start + chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_sentences(text: str, max_sentences: int = 5) -> list[str]:
    """Group consecutive sentences into windows of up to max_sentences each."""
    sentences = _split_sentences(text)
    return [
        " ".join(sentences[i : i + max_sentences])
        for i in range(0, len(sentences), max_sentences)
    ]


def chunk_semantic(text: str, embedder, breakpoint_percentile: float = 95.0) -> list[str]:
    """
    Split into sentences, embed each one, and cut after any sentence whose
    distance to the next is an outlier (above breakpoint_percentile of the
    distance distribution in this document). Runs of sentences between cuts
    become chunks.
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return sentences

    embeddings = embedder.embed_documents(sentences)  # (n, d)
    a, b = embeddings[:-1], embeddings[1:]
    sims = np.sum(a * b, axis=-1) / (
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + EPS
    )
    distances = 1 - sims  # (n-1,), distances[i] = dist(sentence[i], sentence[i+1])

    threshold = np.percentile(distances, breakpoint_percentile)
    breakpoints = set(np.where(distances > threshold)[0].tolist())  # split after sentence i

    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        if (i - 1) in breakpoints:
            chunks.append(" ".join(current))
            current = []
        current.append(sentences[i])
    chunks.append(" ".join(current))
    return chunks
