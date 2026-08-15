"""
Same benchmark protocol as basic_search.py (recall@k, build time, search
latency, memory footprint, ours vs. faiss), but on real sentence embeddings
instead of random Gaussian vectors. Random vectors have no exploitable
structure, so every approximate index looks equally bad regardless of
implementation quality; this script indexes real text so recall differences
actually mean something, and prints a real query's retrieved text so you can
eyeball whether the results make sense.

Corpus: "Pride and Prejudice" from Project Gutenberg, split into ~3-sentence
chunks and embedded with the project's own OllamaEmbedder (nomic-embed-text,
768-dim). Requires `ollama serve` running with nomic-embed-text pulled (see
README). The book text and its embeddings are cached under examples/data/
(gitignored) so repeat runs don't re-download or re-embed.

Queries are hand-written questions about the book's actual plot, held out
from the indexed chunks (not sampled from them), so recall@k measures how
well each index approximates FlatIndex's exact answer to a real question --
same ground-truth convention as basic_search.py.

nomic-embed-text is tuned for cosine similarity, so every index here uses
metric="cosine". faiss has no cosine metric type for these index classes;
the standard equivalent is L2-normalizing vectors and using inner product,
which is what the faiss side below does.
"""

import json
import re
import urllib.request
from pathlib import Path

import faiss
import numpy as np
from bench_utils import bench, bench_faiss, print_row, recall_at_k
from tqdm import tqdm

from rag.chunker import chunk_sentences
from rag.embedder import OllamaEmbedder
from vectordb.index import FlatIndex, HNSWIndex, IVFIndex, PQIndex

DATA_DIR = Path(__file__).parent / "data"
BOOK_URL = "https://www.gutenberg.org/files/1342/1342-0.txt"
BOOK_PATH = DATA_DIR / "pride_and_prejudice.txt"
CACHE_CHUNKS_PATH = DATA_DIR / "pride_and_prejudice_chunks.json"
CACHE_EMBED_PATH = DATA_DIR / "pride_and_prejudice_embeddings.npy"

START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK 1342 ***"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK 1342 ***"

K = 10
EMBED_BATCH_SIZE = 200

# sized for a corpus of ~2k chunks -- see README/basic_search.py for why the
# points-per-cluster ratio matters (faiss itself warns below ~40 points per
# centroid/codeword; nlist=32 and Ks=32 keep every cluster comfortably above
# that on a dataset this size)
IVF_NLIST, IVF_NPROBE = 32, 6
PQ_M, PQ_KS = 8, 32
HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH = 16, 200, 50

QUERIES = [
    "What does Mr. Darcy think of Elizabeth Bennet when they first meet?",
    "How does Mr. Collins propose marriage to Elizabeth?",
    "What is Mrs. Bennet's opinion about her daughters getting married?",
    "Describe the friendship between Jane Bennet and Mr. Bingley.",
    "What does Lady Catherine de Bourgh think of Elizabeth?",
    "How does Elizabeth react to Mr. Darcy's first proposal?",
    "What role does Mr. Wickham play in Lydia's story?",
    "What advice does Mr. Bennet give his daughters?",
    "How is Charlotte Lucas's marriage to Mr. Collins described?",
    "What happens at the ball at Netherfield?",
    "How does Elizabeth's opinion of Mr. Darcy change by the end of the novel?",
    "What does Mr. Bingley's sister think of the Bennet family?",
]


def _load_chunks() -> list[str]:
    """Download the book if needed, strip Project Gutenberg's header/footer,
    collapse whitespace, and split into ~3-sentence chunks."""
    if not BOOK_PATH.exists():
        DATA_DIR.mkdir(exist_ok=True)
        print(f"Downloading {BOOK_URL} ...")
        urllib.request.urlretrieve(BOOK_URL, BOOK_PATH)

    text = BOOK_PATH.read_text(encoding="utf-8")
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    body = text[start:end].split("\n", 1)[1] if start != -1 else text
    body = re.sub(r"\s+", " ", body)
    return chunk_sentences(body, max_sentences=3)


def _load_or_embed(chunks: list[str]) -> np.ndarray:
    """Embed all chunks via the local Ollama model, batched, caching to disk
    so re-running this script doesn't re-embed ~2000 chunks every time."""
    if CACHE_EMBED_PATH.exists() and CACHE_CHUNKS_PATH.exists():
        if json.loads(CACHE_CHUNKS_PATH.read_text()) == chunks:
            return np.load(CACHE_EMBED_PATH)

    embedder = OllamaEmbedder(model="nomic-embed-text")
    batches = [
        embedder.embed_documents(chunks[i : i + EMBED_BATCH_SIZE])
        for i in tqdm(range(0, len(chunks), EMBED_BATCH_SIZE), desc="embedding")
    ]
    vectors = np.vstack(batches)

    DATA_DIR.mkdir(exist_ok=True)
    np.save(CACHE_EMBED_PATH, vectors)
    CACHE_CHUNKS_PATH.write_text(json.dumps(chunks))
    return vectors


def main() -> None:
    chunks = _load_chunks()
    print(f"{len(chunks)} chunks from Pride and Prejudice\n")

    vectors = _load_or_embed(chunks)
    ids = np.arange(len(chunks))
    D = vectors.shape[1]

    embedder = OllamaEmbedder(model="nomic-embed-text")
    queries = np.stack([embedder.embed_query(q) for q in QUERIES])

    vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
    queries_f32 = np.ascontiguousarray(queries, dtype=np.float32)
    faiss.normalize_L2(vectors_f32)
    faiss.normalize_L2(queries_f32)

    header = (
        f"{'Index':<8} {'Recall@10':>10} {'Build (s)':>10} {'Search (ms)':>12} {'Memory (MB)':>12}"
    )

    print("Ours")
    print(header)

    flat = FlatIndex(metric="cosine")
    flat_results, flat_build, flat_search, flat_mem = bench(flat, vectors, ids, queries, K)
    print_row("Flat", 1.000, flat_build, flat_search, flat_mem)

    ivf = IVFIndex(nlist=IVF_NLIST, nprobe=IVF_NPROBE, metric="cosine")
    ivf_results, ivf_build, ivf_search, ivf_mem = bench(
        ivf, vectors, ids, queries, K, train_sample=vectors
    )
    ivf_recall = np.mean([recall_at_k(r, t) for r, t in zip(ivf_results, flat_results)])
    print_row("IVF", ivf_recall, ivf_build, ivf_search, ivf_mem)

    pq = PQIndex(M=PQ_M, Ks=PQ_KS, metric="cosine")
    pq_results, pq_build, pq_search, pq_mem = bench(
        pq, vectors, ids, queries, K, train_sample=vectors
    )
    pq_recall = np.mean([recall_at_k(r, t) for r, t in zip(pq_results, flat_results)])
    print_row("PQ", pq_recall, pq_build, pq_search, pq_mem)

    hnsw = HNSWIndex(
        M=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION, ef_search=HNSW_EF_SEARCH, metric="cosine"
    )
    hnsw_results, hnsw_build, hnsw_search, hnsw_mem = bench(hnsw, vectors, ids, queries, K)
    hnsw_recall = np.mean([recall_at_k(r, t) for r, t in zip(hnsw_results, flat_results)])
    print_row("HNSW", hnsw_recall, hnsw_build, hnsw_search, hnsw_mem)

    # --- faiss reference: normalized vectors + inner product, the standard
    # cosine equivalent since faiss has no cosine metric type for these index
    # classes ---
    print("\nfaiss (normalized + inner product for cosine)")
    print(header)

    faiss_flat = faiss.IndexFlatIP(D)
    _, faiss_flat_build, faiss_flat_search, faiss_flat_mem = bench_faiss(
        faiss_flat, vectors_f32, queries_f32, K
    )
    print_row("Flat", 1.000, faiss_flat_build, faiss_flat_search, faiss_flat_mem)

    faiss_ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(D), D, IVF_NLIST, faiss.METRIC_INNER_PRODUCT)
    faiss_ivf_results, faiss_ivf_build, faiss_ivf_search, faiss_ivf_mem = bench_faiss(
        faiss_ivf, vectors_f32, queries_f32, K, train_sample=vectors_f32, nprobe=IVF_NPROBE
    )
    faiss_ivf_recall = np.mean(
        [recall_at_k(r, t) for r, t in zip(faiss_ivf_results, flat_results)]
    )
    print_row("IVF", faiss_ivf_recall, faiss_ivf_build, faiss_ivf_search, faiss_ivf_mem)

    nbits = int(np.log2(PQ_KS))  # e.g. Ks=32 -> nbits=5 (2**5 == 32 centroids/subspace)
    faiss_pq = faiss.IndexPQ(D, PQ_M, nbits, faiss.METRIC_INNER_PRODUCT)
    faiss_pq_results, faiss_pq_build, faiss_pq_search, faiss_pq_mem = bench_faiss(
        faiss_pq, vectors_f32, queries_f32, K, train_sample=vectors_f32
    )
    faiss_pq_recall = np.mean(
        [recall_at_k(r, t) for r, t in zip(faiss_pq_results, flat_results)]
    )
    print_row("PQ", faiss_pq_recall, faiss_pq_build, faiss_pq_search, faiss_pq_mem)

    faiss_hnsw = faiss.IndexHNSWFlat(D, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    faiss_hnsw.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    faiss_hnsw_results, faiss_hnsw_build, faiss_hnsw_search, faiss_hnsw_mem = bench_faiss(
        faiss_hnsw, vectors_f32, queries_f32, K, ef_search=HNSW_EF_SEARCH
    )
    faiss_hnsw_recall = np.mean(
        [recall_at_k(r, t) for r, t in zip(faiss_hnsw_results, flat_results)]
    )
    print_row("HNSW", faiss_hnsw_recall, faiss_hnsw_build, faiss_hnsw_search, faiss_hnsw_mem)

    # --- sanity check: does a real query actually retrieve relevant text? ---
    print("\n--- sanity check: top-3 chunks for one query (exact search) ---")
    sample_i = 0
    print(f"Q: {QUERIES[sample_i]}\n")
    for rank, chunk_id in enumerate(flat_results[sample_i][:3], start=1):
        print(f"[{rank}] {chunks[chunk_id]}\n")


if __name__ == "__main__":
    main()
