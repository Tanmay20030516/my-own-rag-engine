"""
Tests for vectordb/index/flat.py.

FlatIndex is the correctness baseline: search results are checked against a
naive brute-force loop over distance.py's scalar functions.
"""

import numpy as np
import pytest

from vectordb import distance
from vectordb.index import FlatIndex


@pytest.fixture
def rng():
    return np.random.RandomState(0)


def _naive_topk(X, ids, query, k, scalar_fn):
    dists = np.array([scalar_fn(query, x) for x in X])
    order = np.argsort(dists)[:k]
    return dists[order], ids[order]


@pytest.mark.parametrize(
    "metric,scalar_fn",
    [
        ("l2", distance.l2_squared),
        ("cosine", distance.cosine_distance),
        ("ip", distance.inner_product),
    ],
)
def test_search_matches_naive_loop(rng, metric, scalar_fn):
    X = rng.randn(50, 8)
    ids = np.arange(50)
    query = rng.randn(8)

    index = FlatIndex(metric=metric)
    index.add(X, ids)
    got_dists, got_ids = index.search(query, k=5)

    exp_dists, exp_ids = _naive_topk(X, ids, query, 5, scalar_fn)
    assert np.array_equal(got_ids, exp_ids)
    assert np.allclose(got_dists, exp_dists, atol=1e-6)


def test_search_results_sorted_ascending(rng):
    X = rng.randn(30, 4)
    index = FlatIndex(metric="l2")
    index.add(X)
    dists, _ = index.search(rng.randn(4), k=10)
    assert np.all(np.diff(dists) >= 0)


def test_add_without_ids_assigns_sequential_ids(rng):
    index = FlatIndex()
    index.add(rng.randn(3, 4))
    index.add(rng.randn(2, 4))
    dists, ids = index.search(rng.randn(4), k=5)
    assert set(ids.tolist()) == {0, 1, 2, 3, 4}


def test_add_with_explicit_ids_preserved(rng):
    index = FlatIndex()
    X = rng.randn(3, 4)
    custom_ids = np.array([100, 200, 300])
    index.add(X, custom_ids)
    _, ids = index.search(X[0], k=1)
    assert ids[0] == 100


def test_add_mismatched_ids_length_raises(rng):
    index = FlatIndex()
    with pytest.raises(ValueError):
        index.add(rng.randn(3, 4), ids=np.array([1, 2]))


def test_search_k_larger_than_n_returns_all(rng):
    index = FlatIndex()
    index.add(rng.randn(3, 4))
    dists, ids = index.search(rng.randn(4), k=100)
    assert len(ids) == 3
    assert len(dists) == 3


def test_search_on_empty_index_returns_empty(rng):
    index = FlatIndex()
    dists, ids = index.search(rng.randn(4), k=5)
    assert len(dists) == 0
    assert len(ids) == 0


def test_search_finds_exact_match_with_zero_distance(rng):
    X = rng.randn(10, 6)
    index = FlatIndex(metric="l2")
    index.add(X)
    dists, ids = index.search(X[3], k=1)
    assert ids[0] == 3
    assert dists[0] == pytest.approx(0.0, abs=1e-8)


def test_train_is_a_noop(rng):
    index = FlatIndex()
    index.train(rng.randn(5, 4))  # should not raise or change state
    assert index._vectors is None


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        FlatIndex(metric="not-a-metric")


def test_multiple_add_calls_accumulate(rng):
    index = FlatIndex()
    index.add(rng.randn(5, 4))
    index.add(rng.randn(7, 4))
    dists, ids = index.search(rng.randn(4), k=12)
    assert len(ids) == 12
    assert index._vectors.shape[0] == 12
