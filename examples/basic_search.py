"""
Benchmark demo: add N random D-dim vectors (see constants below) to all four
index types, run NUM_QUERIES queries against each, and report recall@10
(against FlatIndex's exact search as ground truth), average query latency,
and memory footprint. Then repeat the IVF/PQ/HNSW benchmarks against faiss,
using matching hyperparameters, so our from-scratch implementations have a
reference point to compare against.

IVF trains on a random subsample of the data rather than the full set --
standard ANN practice (FAISS does the same): nlist centroids don't need to
see every point to place stable centroids, and a smaller fit set keeps
k-means fast. add() still indexes and search() still queries the complete
dataset -- only the clustering fit is subsampled.

PQ trains on the full dataset instead. Its codebooks use Ks=256 centroids
*per subspace* -- with too small a training sample, you get well under one
point per centroid on average, so the codebook doesn't generalize to the
rest of the dataset and recall craters. KMeans is now vectorized (see
vectordb/kmeans.py), so training all M=8 subspace codebooks on the full set
is cheap enough to just do.

N/D here (100k x 512) are sized to stay well within a 16GB machine's RAM and
finish in a few minutes -- HNSW's build is a pure per-node Python loop (the
algorithm is inherently incremental, not vectorizable the way k-means was),
so it dominates runtime and scales roughly linearly with N.
"""

import faiss
import numpy as np
from bench_utils import bench, bench_faiss, print_row, recall_at_k

from vectordb.index import FlatIndex, HNSWIndex, IVFIndex, PQIndex

N, D, K = 100_000, 512, 10
NUM_QUERIES = 10
IVF_TRAIN_SAMPLE = 20_000

IVF_NLIST, IVF_NPROBE = 512, 16
PQ_M, PQ_KS = 8, 256
HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH = 16, 200, 50


def main() -> None:
    rng = np.random.RandomState(0)
    vectors = rng.randn(N, D)
    ids = np.arange(N)
    queries = rng.randn(NUM_QUERIES, D)
    vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
    queries_f32 = np.ascontiguousarray(queries, dtype=np.float32)

    print(f"{N} vectors, dim={D}, {NUM_QUERIES} queries, k={K}\n")

    header = (
        f"{'Index':<8} {'Recall@10':>10} {'Build (s)':>10} {'Search (ms)':>12} {'Memory (MB)':>12}"
    )

    print("Ours")
    print(header)

    flat = FlatIndex(metric="l2")
    flat_results, flat_build, flat_search, flat_mem = bench(flat, vectors, ids, queries, K)
    print_row("Flat", 1.000, flat_build, flat_search, flat_mem)

    ivf = IVFIndex(nlist=IVF_NLIST, nprobe=IVF_NPROBE, metric="l2")
    ivf_results, ivf_build, ivf_search, ivf_mem = bench(
        ivf, vectors, ids, queries, K, train_sample=vectors[:IVF_TRAIN_SAMPLE]
    )
    ivf_recall = np.mean([recall_at_k(r, t) for r, t in zip(ivf_results, flat_results)])
    print_row("IVF", ivf_recall, ivf_build, ivf_search, ivf_mem)

    pq = PQIndex(M=PQ_M, Ks=PQ_KS, metric="l2")
    pq_results, pq_build, pq_search, pq_mem = bench(
        pq, vectors, ids, queries, K, train_sample=vectors
    )
    pq_recall = np.mean([recall_at_k(r, t) for r, t in zip(pq_results, flat_results)])
    print_row("PQ", pq_recall, pq_build, pq_search, pq_mem)

    hnsw = HNSWIndex(
        M=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION, ef_search=HNSW_EF_SEARCH, metric="l2"
    )
    hnsw_results, hnsw_build, hnsw_search, hnsw_mem = bench(hnsw, vectors, ids, queries, K)
    hnsw_recall = np.mean([recall_at_k(r, t) for r, t in zip(hnsw_results, flat_results)])
    print_row("HNSW", hnsw_recall, hnsw_build, hnsw_search, hnsw_mem)

    # --- faiss reference, same hyperparameters, same ground truth (flat_results) ---
    print("\nfaiss (same params)")
    print(header)

    faiss_flat = faiss.IndexFlatL2(D)
    _, faiss_flat_build, faiss_flat_search, faiss_flat_mem = bench_faiss(
        faiss_flat, vectors_f32, queries_f32, K
    )
    print_row("Flat", 1.000, faiss_flat_build, faiss_flat_search, faiss_flat_mem)

    faiss_ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(D), D, IVF_NLIST, faiss.METRIC_L2)
    faiss_ivf_results, faiss_ivf_build, faiss_ivf_search, faiss_ivf_mem = bench_faiss(
        faiss_ivf,
        vectors_f32,
        queries_f32,
        K,
        train_sample=vectors_f32[:IVF_TRAIN_SAMPLE],
        nprobe=IVF_NPROBE,
    )
    faiss_ivf_recall = np.mean(
        [recall_at_k(r, t) for r, t in zip(faiss_ivf_results, flat_results)]
    )
    print_row("IVF", faiss_ivf_recall, faiss_ivf_build, faiss_ivf_search, faiss_ivf_mem)

    # nbits=8 -> 2**8 = 256 centroids per subspace, matching our Ks=256
    faiss_pq = faiss.IndexPQ(D, PQ_M, 8, faiss.METRIC_L2)
    faiss_pq_results, faiss_pq_build, faiss_pq_search, faiss_pq_mem = bench_faiss(
        faiss_pq, vectors_f32, queries_f32, K, train_sample=vectors_f32
    )
    faiss_pq_recall = np.mean(
        [recall_at_k(r, t) for r, t in zip(faiss_pq_results, flat_results)]
    )
    print_row("PQ", faiss_pq_recall, faiss_pq_build, faiss_pq_search, faiss_pq_mem)

    faiss_hnsw = faiss.IndexHNSWFlat(D, HNSW_M, faiss.METRIC_L2)
    faiss_hnsw.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    faiss_hnsw_results, faiss_hnsw_build, faiss_hnsw_search, faiss_hnsw_mem = bench_faiss(
        faiss_hnsw, vectors_f32, queries_f32, K, ef_search=HNSW_EF_SEARCH
    )
    faiss_hnsw_recall = np.mean(
        [recall_at_k(r, t) for r, t in zip(faiss_hnsw_results, flat_results)]
    )
    print_row("HNSW", faiss_hnsw_recall, faiss_hnsw_build, faiss_hnsw_search, faiss_hnsw_mem)


if __name__ == "__main__":
    main()
