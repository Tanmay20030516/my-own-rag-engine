"""
Hyperparameter sweep that produces the raw numbers behind report.md.

Indexes real sentence embeddings (see corpus.py) with every index type, ours
and faiss side by side, across a grid of hyperparameters, and records recall@k,
build time, search latency, and memory footprint for each configuration.
Results are appended to a flat JSON list at sweep_results.json, which
plots.py reads to render the figures under plots/.

Methodology
-----------
* Ground truth is our own FlatIndex (exact search) over the indexed set, so
  recall@k measures how faithfully each approximate index reproduces the exact
  answer. Both implementations are scored against that same ground truth, which
  makes ours-vs-faiss numbers directly comparable.
* Queries are held-out chunks, not chunks in the index: the corpus is randomly
  split (seeded) into an indexed set and N_QUERIES query vectors. Held-out real
  embeddings give a large enough query set for smooth recall curves, which a
  dozen hand-written questions can't (real_search.py covers that qualitative
  case instead).
* numpy's global RNG is reseeded before every build of ours, because k-means++
  init and HNSW's level assignment both draw from it -- without that, repeat
  runs of the same configuration differ enough to muddy a sweep. faiss seeds
  its own clustering RNG internally and is already deterministic.
* nprobe (IVF) and ef_search (HNSW) only affect search, so those sweeps reuse
  one built index and mutate the attribute instead of rebuilding; the recorded
  build time is the shared build's.
* nomic-embed-text is tuned for cosine, so ours uses metric="cosine". faiss has
  no cosine metric for these index classes, so the faiss side uses the standard
  equivalent: L2-normalized vectors with inner product.
* PQ is the exception to that last point. Our PQIndex implements cosine by
  normalizing and then using *L2* within each subspace (see pq.py), so the
  apples-to-apples faiss config is METRIC_L2 over the same normalized vectors,
  not METRIC_INNER_PRODUCT. The distinction is not cosmetic: PQ codebooks are
  trained to minimize L2 reconstruction error, and scoring them with an L2 ADC
  table measurably beats scoring them with an IP one even though the two are
  rank-equivalent on *exact* distances. The IP variant is recorded alongside
  (impl "faiss-ip") so report.md can document the gap instead of accidentally
  flattering our implementation by handing faiss the worse configuration.
"""

import json
import time
from pathlib import Path

import faiss
import numpy as np
from bench_utils import faiss_index_bytes, our_index_bytes, recall_at_k
from corpus import dedupe, load_chunks, load_or_embed

from vectordb.index import FlatIndex, HNSWIndex, IVFIndex, PQIndex

MAX_SENTENCES = 1  # sentence-level chunks -> ~5.9k vectors, enough to train Ks=128
TAG = f"pnp_s{MAX_SENTENCES}"
N_QUERIES = 300
K = 10
SEED = 0

RESULTS_PATH = Path(__file__).resolve().parents[1] / "sweep_results.json"

# defaults for the headline table; each sweep varies one axis off these
DEF_IVF = {"nlist": 64, "nprobe": 8}
DEF_PQ = {"M": 8, "Ks": 64}
DEF_HNSW = {"M": 16, "ef_construction": 200, "ef_search": 50}

records: list[dict] = []


def log(msg: str) -> None:
    print(msg, flush=True)


def add_record(family, impl, sweep, params, recall, build_s, search_ms, memory_mb) -> None:
    records.append(
        {
            "family": family,
            "impl": impl,
            "sweep": sweep,
            "params": params,
            "recall": float(recall),
            "build_s": float(build_s),
            "search_ms": float(search_ms),
            "memory_mb": float(memory_mb),
        }
    )
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    log(
        f"  {impl:<5} {family:<4} {param_str:<40} "
        f"recall={recall:.3f} build={build_s:7.2f}s search={search_ms:6.3f}ms mem={memory_mb:7.2f}MB"
    )


# --------------------------------------------------------------------------
# our implementations
# --------------------------------------------------------------------------
def build_ours(factory, vectors, ids, train_vectors):
    """Construct + train + add, timed. Reseeds the global RNG first so the
    k-means++ / HNSW-level draws are identical across configurations."""
    np.random.seed(SEED)
    t0 = time.perf_counter()
    index = factory()
    if train_vectors is not None:
        index.train(train_vectors)
    index.add(vectors, ids)
    return index, time.perf_counter() - t0


def search_ours(index, queries, k):
    t0 = time.perf_counter()
    results = [index.search(q, k)[1] for q in queries]
    return results, (time.perf_counter() - t0) / len(queries) * 1000


def score_ours(index, queries, truth, k):
    results, search_ms = search_ours(index, queries, k)
    recall = float(np.mean([recall_at_k(r, t) for r, t in zip(results, truth)]))
    return recall, search_ms, our_index_bytes(index) / (1024**2)


# --------------------------------------------------------------------------
# faiss counterparts
# --------------------------------------------------------------------------
def build_faiss(factory, vectors_f32, train_f32):
    t0 = time.perf_counter()
    index = factory()
    if train_f32 is not None:
        index.train(train_f32)
    index.add(vectors_f32)
    return index, time.perf_counter() - t0


def score_faiss(index, queries_f32, truth, k):
    t0 = time.perf_counter()
    results = [index.search(q.reshape(1, -1), k)[1][0] for q in queries_f32]
    search_ms = (time.perf_counter() - t0) / len(queries_f32) * 1000
    recall = float(np.mean([recall_at_k(r, t) for r, t in zip(results, truth)]))
    return recall, search_ms, faiss_index_bytes(index) / (1024**2)


def main() -> None:
    chunks = load_chunks(max_sentences=MAX_SENTENCES)
    vectors_all = load_or_embed(chunks, tag=TAG)
    n_raw = len(chunks)
    chunks, vectors_all = dedupe(chunks, vectors_all)
    n_all, D = vectors_all.shape

    # seeded split: hold out N_QUERIES real embeddings as the query set
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n_all)
    query_rows, index_rows = perm[:N_QUERIES], perm[N_QUERIES:]
    vectors = np.ascontiguousarray(vectors_all[index_rows])
    queries = np.ascontiguousarray(vectors_all[query_rows])
    ids = np.arange(vectors.shape[0])  # sequential, matching faiss's implicit ids

    vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
    queries_f32 = np.ascontiguousarray(queries, dtype=np.float32)
    faiss.normalize_L2(vectors_f32)
    faiss.normalize_L2(queries_f32)

    log(
        f"corpus: {n_all} chunks (max_sentences={MAX_SENTENCES}, "
        f"{n_raw - n_all} duplicates dropped), dim={D}\n"
        f"indexed: {vectors.shape[0]}   held-out queries: {N_QUERIES}   k={K}\n"
    )

    # ---------------- ground truth: our exact search ----------------
    log("Flat (exact, ground truth)")
    flat, flat_build = build_ours(lambda: FlatIndex(metric="cosine"), vectors, ids, None)
    truth, flat_search_ms = search_ours(flat, queries, K)
    add_record(
        "flat", "ours", "default", {}, 1.0, flat_build, flat_search_ms,
        our_index_bytes(flat) / (1024**2),
    )

    faiss_flat, faiss_flat_build = build_faiss(lambda: faiss.IndexFlatIP(D), vectors_f32, None)
    r, s, m = score_faiss(faiss_flat, queries_f32, truth, K)
    add_record("flat", "faiss", "default", {}, r, faiss_flat_build, s, m)

    # ---------------- IVF: nprobe (search-only, one build) ----------------
    nlist = DEF_IVF["nlist"]
    log(f"\nIVF nprobe sweep (nlist={nlist})")
    ivf, ivf_build = build_ours(
        lambda: IVFIndex(nlist=nlist, nprobe=1, metric="cosine"), vectors, ids, vectors
    )
    faiss_ivf, faiss_ivf_build = build_faiss(
        lambda: faiss.IndexIVFFlat(faiss.IndexFlatIP(D), D, nlist, faiss.METRIC_INNER_PRODUCT),
        vectors_f32,
        vectors_f32,
    )
    for nprobe in (1, 2, 4, 8, 16, 32, 64):
        params = {"nlist": nlist, "nprobe": nprobe}
        ivf.nprobe = nprobe
        r, s, m = score_ours(ivf, queries, truth, K)
        add_record("ivf", "ours", "nprobe", params, r, ivf_build, s, m)

        faiss_ivf.nprobe = nprobe
        r, s, m = score_faiss(faiss_ivf, queries_f32, truth, K)
        add_record("ivf", "faiss", "nprobe", params, r, faiss_ivf_build, s, m)

    # ---------------- IVF: nlist (rebuild each) ----------------
    log(f"\nIVF nlist sweep (nprobe={DEF_IVF['nprobe']})")
    for nl in (16, 32, 64, 128, 256):
        nprobe = min(DEF_IVF["nprobe"], nl)
        params = {"nlist": nl, "nprobe": nprobe}

        idx, build_s = build_ours(
            lambda nl=nl, nprobe=nprobe: IVFIndex(nlist=nl, nprobe=nprobe, metric="cosine"),
            vectors, ids, vectors,
        )
        r, s, m = score_ours(idx, queries, truth, K)
        add_record("ivf", "ours", "nlist", params, r, build_s, s, m)

        fidx, fbuild = build_faiss(
            lambda nl=nl: faiss.IndexIVFFlat(
                faiss.IndexFlatIP(D), D, nl, faiss.METRIC_INNER_PRODUCT
            ),
            vectors_f32, vectors_f32,
        )
        fidx.nprobe = nprobe
        r, s, m = score_faiss(fidx, queries_f32, truth, K)
        add_record("ivf", "faiss", "nlist", params, r, fbuild, s, m)

    # ---------------- PQ: Ks (rebuild each) ----------------
    log(f"\nPQ Ks sweep (M={DEF_PQ['M']})")
    M = DEF_PQ["M"]
    for Ks in (16, 32, 64, 128, 256):
        params = {"M": M, "Ks": Ks}

        idx, build_s = build_ours(
            lambda Ks=Ks: PQIndex(M=M, Ks=Ks, metric="cosine"), vectors, ids, vectors
        )
        r, s, m = score_ours(idx, queries, truth, K)
        add_record("pq", "ours", "Ks", params, r, build_s, s, m)

        nbits = int(np.log2(Ks))
        fidx, fbuild = build_faiss(
            lambda nbits=nbits: faiss.IndexPQ(D, M, nbits, faiss.METRIC_L2),
            vectors_f32, vectors_f32,
        )
        r, s, m = score_faiss(fidx, queries_f32, truth, K)
        add_record("pq", "faiss", "Ks", params, r, fbuild, s, m)

        # same codebooks, inner-product ADC table -- recorded to document the gap
        fidx_ip, fbuild_ip = build_faiss(
            lambda nbits=nbits: faiss.IndexPQ(D, M, nbits, faiss.METRIC_INNER_PRODUCT),
            vectors_f32, vectors_f32,
        )
        r, s, m = score_faiss(fidx_ip, queries_f32, truth, K)
        add_record("pq", "faiss-ip", "Ks", params, r, fbuild_ip, s, m)

    # ---------------- PQ: M (rebuild each) ----------------
    log(f"\nPQ M sweep (Ks={DEF_PQ['Ks']})")
    Ks = DEF_PQ["Ks"]
    nbits = int(np.log2(Ks))
    for M_ in (4, 8, 16, 32):
        params = {"M": M_, "Ks": Ks}

        idx, build_s = build_ours(
            lambda M_=M_: PQIndex(M=M_, Ks=Ks, metric="cosine"), vectors, ids, vectors
        )
        r, s, m = score_ours(idx, queries, truth, K)
        add_record("pq", "ours", "M", params, r, build_s, s, m)

        fidx, fbuild = build_faiss(
            lambda M_=M_: faiss.IndexPQ(D, M_, nbits, faiss.METRIC_L2),
            vectors_f32, vectors_f32,
        )
        r, s, m = score_faiss(fidx, queries_f32, truth, K)
        add_record("pq", "faiss", "M", params, r, fbuild, s, m)

        fidx_ip, fbuild_ip = build_faiss(
            lambda M_=M_: faiss.IndexPQ(D, M_, nbits, faiss.METRIC_INNER_PRODUCT),
            vectors_f32, vectors_f32,
        )
        r, s, m = score_faiss(fidx_ip, queries_f32, truth, K)
        add_record("pq", "faiss-ip", "M", params, r, fbuild_ip, s, m)

    # ---------------- HNSW: ef_search (search-only, one build) ----------------
    hM, hefc = DEF_HNSW["M"], DEF_HNSW["ef_construction"]
    log(f"\nHNSW ef_search sweep (M={hM}, ef_construction={hefc})")
    hnsw, hnsw_build = build_ours(
        lambda: HNSWIndex(M=hM, ef_construction=hefc, ef_search=10, metric="cosine"),
        vectors, ids, None,
    )
    faiss_hnsw, faiss_hnsw_build = build_faiss(
        lambda: _faiss_hnsw(D, hM, hefc), vectors_f32, None
    )
    for ef_search in (10, 20, 50, 100, 200, 400):
        params = {"M": hM, "ef_construction": hefc, "ef_search": ef_search}

        hnsw.ef_search = ef_search
        r, s, m = score_ours(hnsw, queries, truth, K)
        add_record("hnsw", "ours", "ef_search", params, r, hnsw_build, s, m)

        faiss_hnsw.hnsw.efSearch = ef_search
        r, s, m = score_faiss(faiss_hnsw, queries_f32, truth, K)
        add_record("hnsw", "faiss", "ef_search", params, r, faiss_hnsw_build, s, m)

    # ---------------- HNSW: M (rebuild each) ----------------
    log(f"\nHNSW M sweep (ef_construction={hefc}, ef_search={DEF_HNSW['ef_search']})")
    efs = DEF_HNSW["ef_search"]
    for M_ in (8, 16, 32):
        params = {"M": M_, "ef_construction": hefc, "ef_search": efs}

        idx, build_s = build_ours(
            lambda M_=M_: HNSWIndex(
                M=M_, ef_construction=hefc, ef_search=efs, metric="cosine"
            ),
            vectors, ids, None,
        )
        r, s, m = score_ours(idx, queries, truth, K)
        add_record("hnsw", "ours", "M", params, r, build_s, s, m)

        fidx, fbuild = build_faiss(lambda M_=M_: _faiss_hnsw(D, M_, hefc), vectors_f32, None)
        fidx.hnsw.efSearch = efs
        r, s, m = score_faiss(fidx, queries_f32, truth, K)
        add_record("hnsw", "faiss", "M", params, r, fbuild, s, m)

    # ---------------- HNSW: ef_construction (rebuild each) ----------------
    log(f"\nHNSW ef_construction sweep (M={hM}, ef_search={efs})")
    for efc in (50, 100, 200, 400):
        params = {"M": hM, "ef_construction": efc, "ef_search": efs}

        idx, build_s = build_ours(
            lambda efc=efc: HNSWIndex(
                M=hM, ef_construction=efc, ef_search=efs, metric="cosine"
            ),
            vectors, ids, None,
        )
        r, s, m = score_ours(idx, queries, truth, K)
        add_record("hnsw", "ours", "ef_construction", params, r, build_s, s, m)

        fidx, fbuild = build_faiss(lambda efc=efc: _faiss_hnsw(D, hM, efc), vectors_f32, None)
        fidx.hnsw.efSearch = efs
        r, s, m = score_faiss(fidx, queries_f32, truth, K)
        add_record("hnsw", "faiss", "ef_construction", params, r, fbuild, s, m)

    meta = {
        "corpus": "Pride and Prejudice (Project Gutenberg #1342)",
        "embed_model": "nomic-embed-text",
        "max_sentences": MAX_SENTENCES,
        "n_chunks_raw": int(n_raw),
        "n_chunks_deduped": int(n_all),
        "n_indexed": int(vectors.shape[0]),
        "n_queries": N_QUERIES,
        "dim": int(D),
        "k": K,
        "seed": SEED,
        "defaults": {"ivf": DEF_IVF, "pq": DEF_PQ, "hnsw": DEF_HNSW},
    }
    RESULTS_PATH.write_text(json.dumps({"meta": meta, "records": records}, indent=2))
    log(f"\nwrote {len(records)} records to {RESULTS_PATH}")


def _faiss_hnsw(D: int, M: int, ef_construction: int):
    """faiss HNSW takes efConstruction as an attribute, not a constructor arg."""
    index = faiss.IndexHNSWFlat(D, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    return index


if __name__ == "__main__":
    main()
