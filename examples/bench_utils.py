"""
Shared benchmarking harness for the examples/ scripts: build an index (ours
or faiss), time build/search, measure memory footprint, and score recall@k
against a ground-truth result set. Used by basic_search.py (synthetic random
vectors) and real_search.py (real sentence embeddings) so both scripts share
one benchmarking protocol instead of duplicating it.
"""

import sys
import time

import faiss
import numpy as np

# attribute names, across all four of our index classes, that hold the
# retained numpy arrays / bookkeeping structures worth counting
_ARRAY_ATTRS = ("_vectors", "_ids", "centroids", "codebooks", "_codes")
_CONTAINER_ATTRS = ("inverted_lists", "layers", "node_level")


def _container_bytes(obj) -> int:
    """Real memory of a nested dict/list/set of Python objects (IVF's inverted
    lists, HNSW's graph adjacency) -- these carry genuine per-object overhead
    that a numpy array.nbytes wouldn't capture."""
    if isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(
            _container_bytes(k) + _container_bytes(v) for k, v in obj.items()
        )
    if isinstance(obj, (list, tuple, set)):
        return sys.getsizeof(obj) + sum(_container_bytes(v) for v in obj)
    return sys.getsizeof(obj)


def our_index_bytes(index) -> int:
    """Exact retained footprint: nbytes of every array the index stores, plus
    real overhead for any dict/list bookkeeping. Deliberately not a process-RSS
    snapshot -- RSS is noisy here because k-means training allocates and frees
    large temporary distance matrices that the allocator doesn't always hand
    back to the OS, which would make PQ (the compression-oriented index) look
    artificially larger than IVF instead of smaller."""
    total = 0
    for name in _ARRAY_ATTRS:
        arr = getattr(index, name, None)
        if arr is not None:
            total += arr.nbytes
    for name in _CONTAINER_ATTRS:
        obj = getattr(index, name, None)
        if obj:
            total += _container_bytes(obj)
    return total


def faiss_index_bytes(index) -> int:
    """Serialized size of a faiss index -- the standard way to measure a faiss
    index's footprint from Python, since its data lives in a C++ heap that
    sys.getsizeof() can't see through."""
    return faiss.serialize_index(index).nbytes


def recall_at_k(retrieved_ids: np.ndarray, truth_ids: np.ndarray) -> float:
    return len(set(retrieved_ids.tolist()) & set(truth_ids.tolist())) / len(truth_ids)


def bench(index, vectors, ids, queries, k, train_sample=None):
    """Build the index (optionally training on a subsample first), then time k-NN
    search over all queries. Memory is our_index_bytes() measured right after build --
    the index's own retained arrays/structures, not a process-wide snapshot."""
    t0 = time.perf_counter()
    if train_sample is not None:
        index.train(train_sample)
    index.add(vectors, ids)
    build_time = time.perf_counter() - t0

    memory_mb = our_index_bytes(index) / (1024**2)

    t0 = time.perf_counter()
    results = [index.search(q, k)[1] for q in queries]
    search_time = (time.perf_counter() - t0) / len(queries)

    return results, build_time, search_time, memory_mb


def bench_faiss(index, vectors_f32, queries_f32, k, train_sample=None, nprobe=None, ef_search=None):
    """Same protocol as bench(), but for a faiss index. faiss assigns sequential
    row ids by default, same as our own `ids = np.arange(N)`, so recall_at_k can
    compare faiss results against our results directly. Memory is faiss_index_bytes()
    (serialized size), the faiss-side equivalent of our_index_bytes()."""
    t0 = time.perf_counter()
    if train_sample is not None:
        index.train(train_sample)
    index.add(vectors_f32)
    build_time = time.perf_counter() - t0

    memory_mb = faiss_index_bytes(index) / (1024**2)

    if nprobe is not None:
        index.nprobe = nprobe
    if ef_search is not None:
        index.hnsw.efSearch = ef_search

    t0 = time.perf_counter()
    results = [index.search(q.reshape(1, -1), k)[1][0] for q in queries_f32]
    search_time = (time.perf_counter() - t0) / len(queries_f32)

    return results, build_time, search_time, memory_mb


def print_row(name, recall, build_time, search_time, memory_mb):
    print(
        f"{name:<8} {recall:>10.3f} {build_time:>10.3f} {search_time * 1000:>12.3f} {memory_mb:>12.1f}"
    )
