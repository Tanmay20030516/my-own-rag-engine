"""
Tests for vectordb/kmeans.py.

These pin down bugs found during code review of the Lloyd's k-means
implementation:
  - fit() breaking after exactly one iteration regardless of max_iter/tol
  - final_assignments / final_centroids going out of sync when the loop
    runs past one iteration
  - predict() collapsing a batch of query points into a single label
  - empty clusters getting a valid reinitialized centroid instead of being
    dropped or left as NaN
"""

import numpy as np
import pytest

from vectordb.kmeans import KMeans


def _three_blob_data(seed=0):
    rng = np.random.RandomState(seed)
    return np.vstack(
        [
            rng.randn(50, 2) + np.array([0, 0]),
            rng.randn(50, 2) + np.array([10, 10]),
            rng.randn(50, 2) + np.array([10, 0]),
        ]
    )


def test_fit_runs_more_than_one_iteration():
    """
    prev_cost starts at 0, and curr_cost is always positive, so
    `prev_cost - curr_cost < tol` is true on the very first check and the
    loop breaks immediately -- max_iter is never honored.

    Well-separated blobs need more than one E/M step to find their true
    centers; a single assignment+update from a random init should not
    already be within `tol` of converged.
    """
    X = _three_blob_data(seed=0)
    true_centers = np.array([[0, 0], [10, 10], [10, 0]])

    km = KMeans(n_clusters=3, max_iter=50, tol=1e-4, init_method="random")
    km.fit(X)

    # each recovered centroid should land near one of the true blob centers
    recovered = np.array(list(km.final_centroids.values()))
    for center in recovered:
        closest = np.min(np.sum((true_centers - center) ** 2, axis=1))
        assert closest < 1.0, (
            f"centroid {center} is not near any true blob center; "
            "fit() likely stopped after a single iteration"
        )


def test_final_assignments_match_final_centroids():
    """
    final_assignments must be the nearest-centroid labeling for
    final_centroids. If fit() breaks with assignments computed against a
    stale/updated set of centroids, most points end up mislabeled.
    """
    X = _three_blob_data(seed=2)

    km = KMeans(n_clusters=3, max_iter=20, tol=1e-6, init_method="random")
    km.fit(X)

    recomputed = km._assign_centroids(
        {k: v for k, v in enumerate(km.final_centroids.values())}
    )
    # recomputed cluster *indices* may be permuted relative to final_centroids'
    # keys if a centroid was dropped, so compare via achieved cost instead of
    # raw label equality.
    cost_stored = km._cost(km.final_assignments, km.final_centroids)
    cost_recomputed = km._cost(recomputed, km.final_centroids)
    assert cost_stored == pytest.approx(cost_recomputed, rel=1e-6), (
        "final_assignments is not the nearest-centroid labeling for "
        "final_centroids -- they were computed against different centroid sets"
    )


def test_fit_converges_below_naive_single_step_cost():
    """
    Regression guard for the max_iter=1 collapse: running with a generous
    max_iter should reach a lower (or equal) cost than running with
    max_iter=1 on the same init.
    """
    X = _three_blob_data(seed=5)

    np.random.seed(42)
    km_short = KMeans(n_clusters=3, max_iter=1, tol=-np.inf, init_method="random")
    km_short.fit(X)
    cost_short = km_short._cost(km_short.final_assignments, km_short.final_centroids)

    np.random.seed(42)
    km_long = KMeans(n_clusters=3, max_iter=50, tol=1e-6, init_method="random")
    km_long.fit(X)
    cost_long = km_long._cost(km_long.final_assignments, km_long.final_centroids)

    assert cost_long <= cost_short + 1e-9


def test_predict_returns_one_label_per_query_point():
    """
    predict() sums over all axes of X, so a batch of query points collapses
    into a single scalar distance and a single label instead of one label
    per row.
    """
    X = _three_blob_data(seed=4)
    km = KMeans(n_clusters=3, max_iter=10, tol=1e-4, init_method="random")
    km.fit(X)

    queries = np.array([[0, 0], [10, 10], [10, 0]], dtype=float)
    labels = km.predict(queries)

    assert hasattr(labels, "__len__") and len(labels) == len(queries), (
        "predict() should return one label per query row, got a single "
        f"value {labels!r} for a batch of {len(queries)} points"
    )


def test_predict_single_point_matches_nearest_centroid():
    X = _three_blob_data(seed=6)
    km = KMeans(n_clusters=3, max_iter=10, tol=1e-4, init_method="random")
    km.fit(X)

    query = np.array([10.0, 10.0])
    label = km.predict(query[None, :])[0]

    centroids = np.array(list(km.final_centroids.values()))
    keys = list(km.final_centroids.keys())
    expected_key = keys[np.argmin(np.sum((centroids - query) ** 2, axis=1))]
    assert label == expected_key


def test_compute_mean_reinitializes_empty_clusters():
    """
    A cluster id with zero assigned points must still get a valid (non-NaN)
    centroid -- reinitialized to the farthest point from its own cluster's
    centroid -- rather than being dropped or left as NaN.
    """
    X = _three_blob_data(seed=1)
    km = KMeans(n_clusters=3, max_iter=1, tol=1e-4)
    km.X_train = X
    km.num_samples = X.shape[0]

    assignments = np.array([0, 1] * (X.shape[0] // 2))  # cluster id 2 never used
    means = km._compute_mean(assignments)

    assert set(means.keys()) == {0, 1, 2}
    for centroid in means.values():
        assert np.all(np.isfinite(centroid))

    # the reinitialized centroid must be an actual data point (the one
    # farthest from its assigned cluster's centroid), not a fabricated value
    assert np.any(np.all(np.isclose(X, means[2]), axis=1))


def test_kmeans_pp_init_returns_requested_number_of_centroids():
    X = _three_blob_data(seed=3)
    km = KMeans(n_clusters=3, max_iter=10, tol=1e-4, init_method="kmeans++")
    km.X_train = X
    km.num_samples, km.num_feat = X.shape

    centroids = km._init_centroids()
    assert centroids.shape == (3, 2)
