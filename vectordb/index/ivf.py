"""
IVFIndex — inverted-file ANN index.

train() clusters the training vectors into nlist centroids with k-means.
add() assigns each vector to its nearest centroid's inverted list. search()
probes only the nprobe centroids nearest the query and scans just those
lists exactly, instead of the whole dataset (see FlatIndex).

nprobe trades recall for speed: nprobe == nlist degrades to exact search;
nprobe == 1 is fastest but can miss neighbors whose true nearest centroid
wasn't among the top nprobe.
"""

from typing import Literal

import numpy as np

from vectordb import distance
from vectordb.index.base import VectorIndex
from vectordb.kmeans import KMeans

Metric = Literal["l2", "cosine", "ip"]

_PAIRWISE_FUNCS = {
    "l2": distance.pairwise_l2,
    "cosine": distance.pairwise_cosine,
    "ip": distance.pairwise_inner_product,
}


class IVFIndex(VectorIndex):

    def __init__(
        self,
        nlist: int = 100,
        nprobe: int = 10,
        metric: Metric = "l2",
    ) -> None:
        if metric not in _PAIRWISE_FUNCS:
            raise ValueError(
                f"unknown metric {metric!r}, expected one of {sorted(_PAIRWISE_FUNCS)}"
            )
        self.nlist = nlist # the number of clusters to create
        self.nprobe = nprobe
        self.metric = metric

        self.centroids: np.ndarray | None = None  # (nlist, d), set by train()
        self.inverted_lists: dict[int, list[int]] | None = None  # centroid_id -> row idx into self._vectors
        self._vectors: np.ndarray | None = None
        self._ids: np.ndarray | None = None

    def train(self, vectors: np.ndarray) -> None:
        """Fit k-means with nlist clusters on vectors; store the centroids and reset empty inverted lists."""
        km = KMeans(n_clusters=self.nlist, max_iter=200, tol=1e-4)
        km.fit(vectors)
        self.centroids = np.stack([km.final_centroids[i] for i in range(self.nlist)])
        # bucketing happens in add(), tied to self._vectors row indices --
        # training-sample assignments don't correspond to anything stored yet
        self.inverted_lists = {i: [] for i in range(self.nlist)}


    def add(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> None:
        """Append vectors to self._vectors/_ids, then bucket each row index into its nearest centroid's list."""
        assert self.centroids is not None, "train() must run before add()"
        assert self.inverted_lists is not None, "train() must run before add()"
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

        # row indices these new vectors will occupy in self._vectors, so the
        # inverted lists can store absolute positions rather than local ones
        row_start = 0 if self._vectors is None else self._vectors.shape[0]

        if self._vectors is None:
            self._vectors = vectors
            self._ids = ids
        else:
            self._vectors = np.vstack([self._vectors, vectors])
            self._ids = np.concat((self._ids, ids))  # type: ignore

        nearest_centroid = np.argmin(
            _PAIRWISE_FUNCS[self.metric](vectors, self.centroids), axis=1
        )
        for offset, centroid_id in enumerate(nearest_centroid):
            self.inverted_lists[int(centroid_id)].append(row_start + offset)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Probe the nprobe nearest centroids, gather their candidates, score exactly, return top-k."""
        if self._vectors is None or self._vectors.shape[0] == 0:
            return (np.array([]), np.array([]))
        assert self.centroids is not None, "train() must run before search()"
        assert self.inverted_lists is not None, "train() must run before search()"
        query = np.asarray(query, dtype=np.float64).reshape(1, -1)

        dists_to_centroids = _PAIRWISE_FUNCS[self.metric](query, self.centroids)[0]
        probe = np.argsort(dists_to_centroids)[:self.nprobe]
        candidate_idx = np.concatenate([self.inverted_lists[int(c)] for c in probe])
        if candidate_idx.shape[0] == 0:
            return (np.array([]), np.array([]))
        k = min(k, candidate_idx.shape[0])
        candidate_vectors = self._vectors[candidate_idx]

        # the usual search in reduced search space
        dists = _PAIRWISE_FUNCS[self.metric](query, candidate_vectors)[0]  # (n,)
        
        top_k = np.argpartition(dists, k - 1)[:k]
        order = np.argsort(dists[top_k])
        top_k = top_k[order]

        # top_k indexes into the candidate subset, not self._vectors/self._ids --
        # map back through candidate_idx to get the real row indices
        real_idx = candidate_idx[top_k]
        return dists[top_k], self._ids[real_idx]  # type: ignore

