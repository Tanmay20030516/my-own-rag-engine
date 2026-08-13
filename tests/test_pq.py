"""
Tests for vectordb/index/pq.py.

PQIndex trades exact search for a big memory saving: it never keeps the
full-precision vectors after add(), just M uint8 codebook indices per
vector. search() reconstructs approximate distances via ADC (asymmetric
distance computation), so unlike IVFIndex there's no setting that makes it
degrade to exact FlatIndex results -- quantization error is always present.
Recall is checked against FlatIndex ground truth instead.

KMeans seeds its centroids off the global numpy RNG (not a local
RandomState), so np.random.seed(...) is called right before every
train() to keep these tests deterministic.
"""

import numpy as np
import pytest

from vectordb.index import FlatIndex, PQIndex


def _random_data(n=40, d=8, seed=0):
    return np.random.RandomState(seed).randn(n, d)


def test_train_sets_codebooks_shape():
    X = _random_data(n=30, d=8, seed=1)
    idx = PQIndex(M=4, Ks=6)
    np.random.seed(1)
    idx.train(X)
    assert idx.codebooks.shape == (4, 6, 2)
    assert idx.sub_dim == 2


def test_vector_dim_not_divisible_by_m_raises():
    X = _random_data(n=10, d=7, seed=2)
    idx = PQIndex(M=3, Ks=4)
    np.random.seed(2)
    with pytest.raises(ValueError):
        idx.train(X)


def test_invalid_ks_raises():
    with pytest.raises(ValueError):
        PQIndex(Ks=0)
    with pytest.raises(ValueError):
        PQIndex(Ks=257)


def test_invalid_m_raises():
    with pytest.raises(ValueError):
        PQIndex(M=0)


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        PQIndex(metric="not-a-metric")


def test_add_before_train_raises():
    idx = PQIndex(M=2, Ks=4)
    with pytest.raises(AssertionError):
        idx.add(_random_data(n=5, d=4, seed=3))


def test_search_before_add_returns_empty():
    idx = PQIndex(M=2, Ks=4)
    np.random.seed(4)
    idx.train(_random_data(n=20, d=4, seed=4))
    dists, ids = idx.search(_random_data(n=1, d=4, seed=5)[0], k=5)
    assert len(dists) == 0
    assert len(ids) == 0


def test_search_on_untrained_empty_index_returns_empty():
    idx = PQIndex(M=2, Ks=4)
    dists, ids = idx.search(_random_data(n=1, d=4, seed=6)[0], k=5)
    assert len(dists) == 0
    assert len(ids) == 0


def test_add_encodes_as_uint8_codes_with_expected_shape():
    X = _random_data(n=20, d=8, seed=7)
    idx = PQIndex(M=4, Ks=6)
    np.random.seed(7)
    idx.train(X)
    idx.add(X)
    assert idx._codes.shape == (20, 4)
    assert idx._codes.dtype == np.uint8
    assert np.all(idx._codes < 6)


def test_add_without_ids_assigns_sequential_ids():
    X = _random_data(n=10, d=4, seed=8)
    idx = PQIndex(M=2, Ks=4)
    np.random.seed(8)
    idx.train(X)
    idx.add(X[:4])
    idx.add(X[4:])
    _, ids = idx.search(X[0], k=10)
    assert set(ids.tolist()) == set(range(10))


def test_add_with_explicit_ids_preserved():
    X = _random_data(n=5, d=4, seed=9)
    idx = PQIndex(M=2, Ks=4)
    np.random.seed(9)
    idx.train(X)
    custom_ids = np.array([100, 200, 300, 400, 500])
    idx.add(X, custom_ids)
    _, ids = idx.search(X[2], k=1)
    assert ids[0] == 300


def test_add_mismatched_ids_length_raises():
    X = _random_data(n=5, d=4, seed=10)
    idx = PQIndex(M=2, Ks=4)
    np.random.seed(10)
    idx.train(X)
    with pytest.raises(ValueError):
        idx.add(X, ids=np.array([1, 2]))


def test_search_results_sorted_ascending():
    X = _random_data(n=40, d=8, seed=11)
    idx = PQIndex(M=4, Ks=6)
    np.random.seed(11)
    idx.train(X)
    idx.add(X)
    dists, _ = idx.search(_random_data(n=1, d=8, seed=12)[0], k=15)
    assert np.all(np.diff(dists) >= 0)


def test_search_k_larger_than_dataset_returns_available_without_crash():
    X = _random_data(n=10, d=4, seed=13)
    idx = PQIndex(M=2, Ks=4)
    np.random.seed(13)
    idx.train(X)
    idx.add(X)
    dists, ids = idx.search(_random_data(n=1, d=4, seed=14)[0], k=50)
    assert len(ids) == len(dists)
    assert len(ids) == 10


def test_exact_match_zero_distance_when_ks_covers_every_point():
    """With one cluster per training point, k-means converges to the
    points themselves, so a query equal to a stored vector should
    reconstruct to distance ~0 and rank first."""
    X = _random_data(n=6, d=4, seed=15)
    idx = PQIndex(M=1, Ks=6, metric="l2")
    np.random.seed(15)
    idx.train(X)
    idx.add(X)
    dists, ids = idx.search(X[3], k=1)
    assert ids[0] == 3
    assert dists[0] == pytest.approx(0.0, abs=1e-6)


def test_recall_against_flat_index_on_separated_clusters():
    """Quantization error means PQ won't match FlatIndex exactly, but on
    well-separated clusters recall shouldn't be far off."""
    rng = np.random.RandomState(16)
    centers = rng.randn(6, 8) * 15
    X = np.vstack([c + rng.randn(20, 8) for c in centers])
    ids = np.arange(X.shape[0])

    pq = PQIndex(M=4, Ks=8, metric="l2")
    np.random.seed(16)
    pq.train(X)
    pq.add(X, ids)

    flat = FlatIndex(metric="l2")
    flat.add(X, ids)

    k = 10
    recalls = []
    for i in range(0, X.shape[0], 12):
        _, fi = flat.search(X[i], k=k)
        _, pi = pq.search(X[i], k=k)
        recalls.append(len(set(fi.tolist()) & set(pi.tolist())) / k)

    assert np.mean(recalls) >= 0.5


def test_ip_metric_ranks_similarly_to_flat():
    rng = np.random.RandomState(17)
    centers = rng.randn(4, 8) * 10
    X = np.vstack([c + rng.randn(15, 8) for c in centers])
    ids = np.arange(X.shape[0])

    pq = PQIndex(M=4, Ks=8, metric="ip")
    np.random.seed(17)
    pq.train(X)
    pq.add(X, ids)

    flat = FlatIndex(metric="ip")
    flat.add(X, ids)

    k = 5
    recalls = []
    for i in range(0, X.shape[0], 10):
        _, fi = flat.search(X[i], k=k)
        _, pi = pq.search(X[i], k=k)
        recalls.append(len(set(fi.tolist()) & set(pi.tolist())) / k)

    assert np.mean(recalls) >= 0.4


def test_cosine_metric_normalizes_before_encoding():
    """Scaling a vector shouldn't change which codebook entries it's
    assigned to under cosine metric, since direction is unaffected."""
    X = _random_data(n=20, d=8, seed=18)
    idx = PQIndex(M=4, Ks=6, metric="cosine")
    np.random.seed(18)
    idx.train(X)
    idx.add(X[:1])
    idx.add(X[:1] * 3.0)
    assert np.array_equal(idx._codes[0], idx._codes[1])


def test_recall_against_flat_index_cosine_metric():
    rng = np.random.RandomState(19)
    centers = rng.randn(5, 8) * 10
    X = np.vstack([c + rng.randn(15, 8) for c in centers])
    ids = np.arange(X.shape[0])

    pq = PQIndex(M=4, Ks=8, metric="cosine")
    np.random.seed(19)
    pq.train(X)
    pq.add(X, ids)

    flat = FlatIndex(metric="cosine")
    flat.add(X, ids)

    k = 5
    recalls = []
    for i in range(0, X.shape[0], 10):
        _, fi = flat.search(X[i], k=k)
        _, pi = pq.search(X[i], k=k)
        recalls.append(len(set(fi.tolist()) & set(pi.tolist())) / k)

    assert np.mean(recalls) >= 0.4
