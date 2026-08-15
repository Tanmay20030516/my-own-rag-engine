"""
Corpus loading + embedding for the benchmark examples.

Downloads "Pride and Prejudice" from Project Gutenberg, strips the
header/footer boilerplate, chunks it with the project's own chunker, and
embeds the chunks with the project's own OllamaEmbedder (nomic-embed-text,
768-dim). Embeddings are cached under examples/data/ (gitignored), keyed by
chunk granularity, so re-running a benchmark doesn't re-download or re-embed.

Shared by real_search.py (single benchmark run) and sweep.py (hyperparameter
sweep for report.md) so both index the identical corpus.
"""

import json
import re
import urllib.request
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rag.chunker import chunk_sentences
from rag.embedder import OllamaEmbedder

DATA_DIR = Path(__file__).parent / "data"
BOOK_URL = "https://www.gutenberg.org/files/1342/1342-0.txt"
BOOK_PATH = DATA_DIR / "pride_and_prejudice.txt"

START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK 1342 ***"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK 1342 ***"

EMBED_BATCH_SIZE = 200
EMBED_MODEL = "nomic-embed-text"


def load_chunks(max_sentences: int = 3) -> list[str]:
    """Download the book if needed, strip Project Gutenberg's header/footer,
    collapse whitespace, and split into chunks of up to max_sentences each."""
    if not BOOK_PATH.exists():
        DATA_DIR.mkdir(exist_ok=True)
        print(f"Downloading {BOOK_URL} ...")
        urllib.request.urlretrieve(BOOK_URL, BOOK_PATH)

    text = BOOK_PATH.read_text(encoding="utf-8")
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    body = text[start:end].split("\n", 1)[1] if start != -1 else text
    body = re.sub(r"\s+", " ", body)
    return chunk_sentences(body, max_sentences=max_sentences)


def load_or_embed(chunks: list[str], tag: str) -> np.ndarray:
    """Embed chunks via the local Ollama model, batched, caching to disk under
    `tag` so re-running doesn't re-embed. The cached chunk list is compared
    against `chunks` so a changed corpus invalidates the cache automatically."""
    chunks_path = DATA_DIR / f"chunks_{tag}.json"
    embed_path = DATA_DIR / f"embeddings_{tag}.npy"

    if embed_path.exists() and chunks_path.exists():
        if json.loads(chunks_path.read_text()) == chunks:
            return np.load(embed_path)

    embedder = OllamaEmbedder(model=EMBED_MODEL)
    batches = [
        embedder.embed_documents(chunks[i : i + EMBED_BATCH_SIZE])
        for i in tqdm(range(0, len(chunks), EMBED_BATCH_SIZE), desc=f"embedding [{tag}]")
    ]
    vectors = np.vstack(batches)

    DATA_DIR.mkdir(exist_ok=True)
    np.save(embed_path, vectors)
    chunks_path.write_text(json.dumps(chunks))
    return vectors


def embed_queries(questions: list[str]) -> np.ndarray:
    """Embed natural-language questions with the query-side task prefix."""
    embedder = OllamaEmbedder(model=EMBED_MODEL)
    return np.stack([embedder.embed_query(q) for q in questions])


def dedupe(chunks: list[str], vectors: np.ndarray) -> tuple[list[str], np.ndarray]:
    """
    Drop rows whose embedding exactly duplicates an earlier row, preserving
    corpus order.

    Sentence-level chunking repeats plenty of short lines verbatim ("Bennet.",
    "Indeed!"), which embed to bit-identical vectors. Duplicates are noise in a
    retrieval corpus, but more importantly they break recall *measurement*: when
    several vectors sit at exactly the same distance from a query, any top-k that
    includes some of the tied set is equally correct, and two implementations
    that break the tie differently look like they disagree. Deduplicating removes
    that false signal at the source, so an exact index scores 1.000 as it should.
    """
    _, first_idx = np.unique(vectors, axis=0, return_index=True)
    keep = np.sort(first_idx)
    return [chunks[i] for i in keep], np.ascontiguousarray(vectors[keep])
