"""
Tests for vectordb/index/ivf.py.

IVFIndex trades exact search for speed by only scanning the nprobe
inverted lists nearest the query. With nprobe == nlist every list is
scanned, so results must match FlatIndex exactly; with nprobe < nlist,
recall is checked against FlatIndex ground truth on well-separated
clusters where losing recall would be an obvious regression.

KMeans seeds its centroids off the global numpy RNG (not a local
RandomState), so np.random.seed(...) is called right before every
train() to keep these tests deterministic.
"""

import numpy as np
import pytest

from vectordb.index import FlatIndex, IVFIndex


def _random_data(n=200, d=8, seed=0):
    return np.random.RandomState(seed).randn(n, d)


def test_train_sets_centroids_and_resets_inverted_lists():
    X = _random_data(n=100, d=6, seed=1)
    idx = IVFIndex(nlist=8, nprobe=8)
    np.random.seed(1)
    idx.train(X)
    assert idx.centroids.shape == (8, 6)
    assert set(idx.inverted_lists.keys()) == set(range(8))
    assert all(v == [] for v in idx.inverted_lists.values())


def test_add_before_train_raises():
    idx = IVFIndex(nlist=4, nprobe=4)
    with pytest.raises(AssertionError):
        idx.add(_random_data(n=5, d=4, seed=2))


def test_search_before_add_returns_empty():
    idx = IVFIndex(nlist=4, nprobe=4)
    np.random.seed(2)
    idx.train(_random_data(n=50, d=4, seed=2))
    dists, ids = idx.search(_random_data(n=1, d=4, seed=3)[0], k=5)
    assert len(dists) == 0
    assert len(ids) == 0


def test_search_on_untrained_empty_index_returns_empty():
    idx = IVFIndex(nlist=4, nprobe=4)
    dists, ids = idx.search(_random_data(n=1, d=4, seed=4)[0], k=5)
    assert len(dists) == 0
    assert len(ids) == 0


def test_add_buckets_each_vector_under_its_nearest_centroid():
    X = _random_data(n=100, d=4, seed=5)
    idx = IVFIndex(nlist=6, nprobe=6)
    np.random.seed(5)
    idx.train(X)
    idx.add(X)

    for centroid_id, row_idxs in idx.inverted_lists.items():
        for row in row_idxs:
            dists_to_all_centroids = np.sum((X[row] - idx.centroids) ** 2, axis=1)
            assert np.argmin(dists_to_all_centroids) == centroid_id


def test_add_bucketing_has_no_duplicates_and_covers_all_vectors():
    """
    Regression: train() must not pre-populate inverted_lists from its own
    training-sample assignments -- add() is solely responsible for
    bucketing, tied to self._vectors row indices, or vectors get bucketed
    twice (once by train(), once by add()) with dangling/duplicate rows.
    """
    X = _random_data(n=60, d=4, seed=6)
    idx = IVFIndex(nlist=5, nprobe=5)
    np.random.seed(6)
    idx.train(X)
    idx.add(X)

    all_bucketed = sorted(i for lst in idx.inverted_lists.values() for i in lst)
    assert all_bucketed == list(range(60))


def test_add_without_ids_assigns_sequential_ids():
    X = _random_data(n=10, d=4, seed=7)
    idx = IVFIndex(nlist=3, nprobe=3)
    np.random.seed(7)
    idx.train(X)
    idx.add(X[:4])
    idx.add(X[4:])
    _, ids = idx.search(X[0], k=10)
    assert set(ids.tolist()) == set(range(10))


def test_add_with_explicit_ids_preserved():
    X = _random_data(n=5, d=4, seed=8)
    idx = IVFIndex(nlist=2, nprobe=2)
    np.random.seed(8)
    idx.train(X)
    custom_ids = np.array([100, 200, 300, 400, 500])
    idx.add(X, custom_ids)
    _, ids = idx.search(X[2], k=1)
    assert ids[0] == 300


def test_add_mismatched_ids_length_raises():
    X = _random_data(n=5, d=4, seed=9)
    idx = IVFIndex(nlist=2, nprobe=2)
    np.random.seed(9)
    idx.train(X)
    with pytest.raises(ValueError):
        idx.add(X, ids=np.array([1, 2]))


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        IVFIndex(metric="not-a-metric")


def test_search_matches_flat_exactly_when_nprobe_equals_nlist():
    X = _random_data(n=200, d=8, seed=10)
    ids = np.arange(200)

    ivf = IVFIndex(nlist=10, nprobe=10, metric="l2")
    np.random.seed(10)
    ivf.train(X)
    ivf.add(X, ids)

    flat = FlatIndex(metric="l2")
    flat.add(X, ids)

    queries = _random_data(n=5, d=8, seed=11)
    for q in queries:
        fd, fi = flat.search(q, k=10)
        id_, ii = ivf.search(q, k=10)
        assert np.array_equal(fi, ii)
        assert np.allclose(fd, id_)


def test_search_results_sorted_ascending():
    X = _random_data(n=100, d=6, seed=12)
    idx = IVFIndex(nlist=6, nprobe=3)
    np.random.seed(12)
    idx.train(X)
    idx.add(X)
    dists, _ = idx.search(_random_data(n=1, d=6, seed=13)[0], k=15)
    assert np.all(np.diff(dists) >= 0)


def test_search_k_larger_than_candidates_returns_available_without_crash():
    """
    Regression: k must be clamped to the number of candidates gathered from
    the probed lists, not the total vector count -- otherwise argpartition
    is asked for a kth index past the end of a small candidate array.
    """
    X = _random_data(n=50, d=4, seed=14)
    idx = IVFIndex(nlist=10, nprobe=1, metric="l2")
    np.random.seed(14)
    idx.train(X)
    idx.add(X)
    dists, ids = idx.search(_random_data(n=1, d=4, seed=15)[0], k=50)
    assert len(ids) == len(dists)
    assert len(ids) <= 50


def test_recall_against_flat_index_on_separated_clusters():
    """nprobe < nlist trades recall for speed -- shouldn't be far off
    FlatIndex's exact top-k on well-separated clusters."""
    rng = np.random.RandomState(16)
    centers = rng.randn(10, 4) * 15
    X = np.vstack([c + rng.randn(30, 4) for c in centers])
    ids = np.arange(X.shape[0])

    ivf = IVFIndex(nlist=10, nprobe=3, metric="l2")
    np.random.seed(16)
    ivf.train(X)
    ivf.add(X, ids)

    flat = FlatIndex(metric="l2")
    flat.add(X, ids)

    k = 10
    recalls = []
    for i in range(0, X.shape[0], 15):
        _, fi = flat.search(X[i], k=k)
        _, ii = ivf.search(X[i], k=k)
        recalls.append(len(set(fi.tolist()) & set(ii.tolist())) / k)

    assert np.mean(recalls) >= 0.6


def test_exact_match_zero_distance_when_nprobe_equals_nlist():
    X = _random_data(n=40, d=5, seed=17)
    idx = IVFIndex(nlist=5, nprobe=5, metric="l2")
    np.random.seed(17)
    idx.train(X)
    idx.add(X)
    dists, ids = idx.search(X[7], k=1)
    assert ids[0] == 7
    assert dists[0] == pytest.approx(0.0, abs=1e-6)
