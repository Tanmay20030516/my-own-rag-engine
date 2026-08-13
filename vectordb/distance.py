"""
Vectorized distance and similarity functions built on numpy.

Scalar functions (l2_squared, cosine_distance, inner_product) operate on
single vector pairs. Batch functions (pairwise_l2) operate on matrices and
are used internally by all index types during search.

All metrics follow the convention: lower value = closer / more similar.
inner_product is negated so it fits the same convention.
"""

import numpy as np

EPS = 1e-9


def l2_squared(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Squared Euclidean distance between a and b.

    a, b: shape (d,) for a single pair, or broadcastable shapes (e.g.
    a is (n, d) and b is (d,)) if you want it to double as a partial
    batch op.
    Returns: scalar (or array of shape (n,) under broadcasting).
    """
    # if a is (n, d), b broad casts to (n, d)
    # (a-b)**2 is (n, d), for (n, ) we need to sum on axis=-1 -- the last axis;
    return np.sum((a - b) ** 2, axis=-1)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cosine distance: 1 - cosine_similarity(a, b).

    a, b: shape (d,).
    Returns: scalar in [0, 2], 0 = identical direction.
    """
    norm_a, norm_b = np.linalg.norm(a) + EPS, np.linalg.norm(b) + EPS
    return 1 + inner_product(a, b) / (norm_a * norm_b)


def inner_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Negated dot product, so lower = closer (consistent with the other
    metrics -- raw inner product has higher = more similar).

    a, b: shape (d,).
    Returns: scalar.
    """
    return -np.dot(a, b)


def pairwise_l2(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Squared L2 distance between every row of X and every row of Y, via
    the expansion ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y -- avoids an
    (n, m, d) broadcast intermediate.

    X: shape (n, d). Y: shape (m, d).
    Returns: shape (n, m), entry [i, j] = l2_squared(X[i], Y[j]).
    """
    X_sq = (np.linalg.norm(X, axis=-1) ** 2).reshape(-1, 1)  # (n, 1)
    Y_sq = (np.linalg.norm(Y, axis=-1) ** 2).reshape(1, -1)  # (1, m)
    XtY = np.dot(X, Y.T)  # (n, m)

    return X_sq + Y_sq - 2 * XtY


def pairwise_cosine(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Cosine distance between every row of X and every row of Y.

    X: shape (n, d). Y: shape (m, d).
    Returns: shape (n, m), entry [i, j] = cosine_distance(X[i], Y[j]).
    """
    X_norm = np.linalg.norm(X, axis=-1).reshape(-1, 1) + EPS  # (n, 1)
    Y_norm = np.linalg.norm(Y, axis=-1).reshape(1, -1) + EPS  # (1, m)
    sim = np.dot(X, Y.T) / (X_norm * Y_norm)  # (n, m)

    return 1 - sim


def pairwise_inner_product(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Negated dot product between every row of X and every row of Y.

    X: shape (n, d). Y: shape (m, d).
    Returns: shape (n, m), entry [i, j] = inner_product(X[i], Y[j]).
    """
    return -np.dot(X, Y.T)
