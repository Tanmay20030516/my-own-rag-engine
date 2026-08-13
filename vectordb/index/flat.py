"""
FlatIndex — exact brute-force nearest neighbor search.

Scores the query against every stored vector with no approximation and no
training step. Slowest index in the package, but exact — the other index
types are verified against it for recall.
"""

from typing import Literal

import numpy as np

from vectordb import distance
from vectordb.index.base import VectorIndex

Metric = Literal["l2", "cosine", "ip"]

_PAIRWISE_FUNCS = {
    "l2": distance.pairwise_l2,
    "cosine": distance.pairwise_cosine,
    "ip": distance.pairwise_inner_product,
}


class FlatIndex(VectorIndex):

    def __init__(self, metric: Metric = "l2") -> None:
        if metric not in _PAIRWISE_FUNCS:
            raise ValueError(
                f"unknown metric {metric!r}, expected one of {sorted(_PAIRWISE_FUNCS)}"
            )
        self.metric = metric
        self._vectors: np.ndarray | None = None
        self._ids: np.ndarray | None = None

    def train(self, vectors: np.ndarray) -> None:
        """No-op: brute-force search needs no fitted structure."""

    def add(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> None:
        """Append vectors (and ids, default sequential) to the flat store."""
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim == 1: # cast for shape like (n, d)
            vectors = vectors[None, :]
        n = vectors.shape[0]

        if ids is None:
            start = 0 if self._ids is None else int(self._ids[-1]) + 1
            ids = np.arange(start, start + n)
        else:
            ids = np.asarray(ids)
            if ids.shape[0] != n:
                raise ValueError(f"got {n} vectors but {ids.shape[0]} ids")

        if self._vectors is None:
            self._vectors = vectors
            self._ids = ids
        else:
            self._vectors = np.vstack([self._vectors, vectors])
            self._ids = np.concat((self._ids, ids))  # type: ignore

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Exact top-k: pairwise distance to every stored vector, argpartition, then sort just the top k."""
        if self._vectors is None or self._vectors.shape[0] == 0:
            return np.array([]), np.array([])

        query = np.asarray(query, dtype=np.float64).reshape(1, -1)
        n = self._vectors.shape[0]
        k = min(k, n)

        dists = _PAIRWISE_FUNCS[self.metric](query, self._vectors)[0]  # (n,)

        top_k = np.argpartition(dists, k - 1)[:k]
        order = np.argsort(dists[top_k])
        top_k = top_k[order]

        return dists[top_k], self._ids[top_k] # type: ignore
