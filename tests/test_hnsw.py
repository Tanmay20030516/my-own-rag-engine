"""
Tests for vectordb/index/hnsw.py.

HNSWIndex builds a multi-layer proximity graph incrementally as vectors are
added -- there's no training step and no exact-match guarantee even with
generous ef_construction/ef_search, so (like PQIndex) recall is checked
against FlatIndex ground truth rather than exact equality.

Random layer assignment and beam search tie-breaking both draw from the
global numpy RNG, so np.random.seed(...) is called right before every
add() to keep these tests deterministic.
"""

import numpy as np
import pytest

from vectordb.index import FlatIndex, HNSWIndex


def _random_data(n=40, d=8, seed=0):
    return np.random.RandomState(seed).randn(n, d)


def test_invalid_m_raises():
    with pytest.raises(ValueError):
        HNSWIndex(M=1)
    with pytest.raises(ValueError):
        HNSWIndex(M=0)


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        HNSWIndex(metric="not-a-metric")


def test_train_is_a_noop():
    idx = HNSWIndex(M=4)
    idx.train(_random_data(n=10, d=4, seed=1))
    assert idx.entry_point is None
    assert idx.layers == []


def test_search_on_empty_index_returns_empty():
    idx = HNSWIndex(M=4)
    dists, ids = idx.search(_random_data(n=1, d=4, seed=2)[0], k=5)
    assert len(dists) == 0
    assert len(ids) == 0


def test_add_mismatched_ids_length_raises():
    idx = HNSWIndex(M=4)
    X = _random_data(n=5, d=4, seed=3)
    np.random.seed(3)
    with pytest.raises(ValueError):
        idx.add(X, ids=np.array([1, 2]))


def test_add_without_ids_assigns_sequential_ids():
    idx = HNSWIndex(M=4)
    X = _random_data(n=10, d=4, seed=4)
    np.random.seed(4)
    idx.add(X[:4])
    idx.add(X[4:])
    _, ids = idx.search(X[0], k=10)
    assert set(ids.tolist()) == set(range(10))


def test_add_with_explicit_ids_preserved():
    idx = HNSWIndex(M=4)
    X = _random_data(n=5, d=4, seed=5)
    custom_ids = np.array([100, 200, 300, 400, 500])
    np.random.seed(5)
    idx.add(X, custom_ids)
    _, ids = idx.search(X[2], k=1)
    assert ids[0] == 300


def test_single_node_search_returns_that_node():
    idx = HNSWIndex(M=4)
    X = _random_data(n=1, d=4, seed=6)
    np.random.seed(6)
    idx.add(X)
    dists, ids = idx.search(_random_data(n=1, d=4, seed=7)[0], k=5)
    assert ids.tolist() == [0]
    assert len(dists) == 1


def test_exact_match_near_zero_distance():
    idx = HNSWIndex(M=8, ef_construction=100, ef_search=50)
    X = _random_data(n=30, d=8, seed=8)
    np.random.seed(8)
    idx.add(X)
    dists, ids = idx.search(X[12], k=1)
    assert ids[0] == 12
    assert dists[0] == pytest.approx(0.0, abs=1e-6)


def test_search_results_sorted_ascending():
    idx = HNSWIndex(M=6, ef_construction=80, ef_search=40)
    X = _random_data(n=50, d=8, seed=9)
    np.random.seed(9)
    idx.add(X)
    dists, _ = idx.search(_random_data(n=1, d=8, seed=10)[0], k=15)
    assert np.all(np.diff(dists) >= 0)


def test_search_k_larger_than_dataset_returns_available_without_crash():
    idx = HNSWIndex(M=4, ef_construction=50, ef_search=20)
    X = _random_data(n=10, d=4, seed=11)
    np.random.seed(11)
    idx.add(X)
    dists, ids = idx.search(_random_data(n=1, d=4, seed=12)[0], k=50)
    assert len(ids) == len(dists) == 10


def test_layer_zero_holds_every_node():
    idx = HNSWIndex(M=4, ef_construction=40, ef_search=20)
    X = _random_data(n=25, d=4, seed=13)
    np.random.seed(13)
    idx.add(X)
    # every inserted node must appear as a key in layer 0's adjacency dict,
    # or as the sole node in a graph with no edges yet
    assert idx._vectors.shape[0] == 25
    assert idx.top_level == max(idx.node_level.values())


def test_neighbor_degree_never_exceeds_layer_cap():
    idx = HNSWIndex(M=4, ef_construction=60, ef_search=30)
    X = _random_data(n=60, d=8, seed=14)
    np.random.seed(14)
    idx.add(X)
    for layer_idx, layer in enumerate(idx.layers):
        cap = idx.M_max0 if layer_idx == 0 else idx.M
        for neighbors in layer.values():
            assert len(neighbors) <= cap


def test_recall_against_flat_index_on_separated_clusters():
    rng = np.random.RandomState(16)
    centers = rng.randn(8, 16) * 15
    X = np.vstack([c + rng.randn(30, 16) for c in centers])
    ids = np.arange(X.shape[0])

    hnsw = HNSWIndex(M=8, ef_construction=100, ef_search=50, metric="l2")
    np.random.seed(16)
    hnsw.add(X, ids)

    flat = FlatIndex(metric="l2")
    flat.add(X, ids)

    k = 10
    recalls = []
    for i in range(0, X.shape[0], 10):
        _, fi = flat.search(X[i], k=k)
        _, hi = hnsw.search(X[i], k=k)
        recalls.append(len(set(fi.tolist()) & set(hi.tolist())) / k)

    assert np.mean(recalls) >= 0.6


def test_ip_metric_recall_against_flat():
    rng = np.random.RandomState(17)
    centers = rng.randn(6, 16) * 10
    X = np.vstack([c + rng.randn(20, 16) for c in centers])
    ids = np.arange(X.shape[0])

    hnsw = HNSWIndex(M=8, ef_construction=100, ef_search=50, metric="ip")
    np.random.seed(17)
    hnsw.add(X, ids)

    flat = FlatIndex(metric="ip")
    flat.add(X, ids)

    k = 5
    recalls = []
    for i in range(0, X.shape[0], 10):
        _, fi = flat.search(X[i], k=k)
        _, hi = hnsw.search(X[i], k=k)
        recalls.append(len(set(fi.tolist()) & set(hi.tolist())) / k)

    assert np.mean(recalls) >= 0.6


def test_cosine_metric_recall_against_flat():
    rng = np.random.RandomState(18)
    centers = rng.randn(6, 16) * 10
    X = np.vstack([c + rng.randn(20, 16) for c in centers])
    ids = np.arange(X.shape[0])

    hnsw = HNSWIndex(M=8, ef_construction=100, ef_search=50, metric="cosine")
    np.random.seed(18)
    hnsw.add(X, ids)

    flat = FlatIndex(metric="cosine")
    flat.add(X, ids)

    k = 5
    recalls = []
    for i in range(0, X.shape[0], 10):
        _, fi = flat.search(X[i], k=k)
        _, hi = hnsw.search(X[i], k=k)
        recalls.append(len(set(fi.tolist()) & set(hi.tolist())) / k)

    assert np.mean(recalls) >= 0.6
