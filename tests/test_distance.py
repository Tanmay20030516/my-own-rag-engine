"""
Tests for vectordb/distance.py.

Covers scalar and batch variants of L2, cosine, and inner product.
Batch pairwise_l2 results are verified against the naive scalar loop.
"""

import numpy as np
import pytest

from vectordb.distance import cosine_distance, inner_product, l2_squared, pairwise_l2


@pytest.fixture
def rng():
    return np.random.RandomState(0)


def test_l2_squared_scalar_matches_naive(rng):
    a = rng.randn(5)
    b = rng.randn(5)
    assert np.isclose(l2_squared(a, b), np.sum((a - b) ** 2))


def test_l2_squared_zero_for_identical_vectors(rng):
    a = rng.randn(5)
    assert l2_squared(a, a) == pytest.approx(0.0)


def test_l2_squared_broadcasts_over_batch(rng):
    A = rng.randn(4, 5)
    b = rng.randn(5)
    naive = np.array([np.sum((A[i] - b) ** 2) for i in range(4)])
    assert np.allclose(l2_squared(A, b), naive)


def test_cosine_distance_matches_naive(rng):
    a = rng.randn(5)
    b = rng.randn(5)
    naive = 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert cosine_distance(a, b) == pytest.approx(naive, abs=1e-6)


def test_cosine_distance_zero_for_identical_direction(rng):
    a = rng.randn(5)
    assert cosine_distance(a, a) == pytest.approx(0.0, abs=1e-6)
    # scaling shouldn't change direction
    assert cosine_distance(a, a * 3.0) == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance_max_for_opposite_vectors(rng):
    a = rng.randn(5)
    assert cosine_distance(a, -a) == pytest.approx(2.0, abs=1e-6)


def test_cosine_distance_handles_zero_vector_without_crash(rng):
    z = np.zeros(5)
    a = rng.randn(5)
    # EPS-guarded norms: should not raise or return NaN/inf
    result = cosine_distance(z, a)
    assert np.isfinite(result)


def test_inner_product_matches_negated_dot(rng):
    a = rng.randn(5)
    b = rng.randn(5)
    assert inner_product(a, b) == pytest.approx(-np.dot(a, b))


def test_pairwise_l2_matches_naive_loop(rng):
    X = rng.randn(6, 5)
    Y = rng.randn(3, 5)
    naive = np.array(
        [[np.sum((X[i] - Y[j]) ** 2) for j in range(3)] for i in range(6)]
    )
    out = pairwise_l2(X, Y)
    assert out.shape == (6, 3)
    assert np.allclose(out, naive, atol=1e-8)


def test_pairwise_l2_self_distance_near_zero(rng):
    """
    The ||x||^2 + ||y||^2 - 2xy expansion can leave tiny floating-point
    residue (including small negatives) on the diagonal instead of an
    exact 0 -- pin the tolerance so a real regression doesn't slip in
    under it.
    """
    X = rng.randn(6, 5)
    diag = np.diag(pairwise_l2(X, X))
    assert np.allclose(diag, 0.0, atol=1e-8)
