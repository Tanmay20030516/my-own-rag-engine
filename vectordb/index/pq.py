"""
PQIndex — product-quantization ANN index.

train() splits vectors into M sub-vectors and fits an independent k-means
codebook (Ks centroids) per subspace. add() encodes each vector as M
codebook indices (uint8) instead of keeping the full-precision vector --
this is where PQ gets its memory saving over Flat/IVF (N*M bytes instead
of N*d floats). search() uses asymmetric distance computation (ADC):
build a (M, Ks) query-to-centroid distance table once per query, then for
every stored code look up and sum M table entries instead of touching the
original dimensionality.

l2 and ip decompose exactly into a sum over subspaces (||q-x||^2 and q.x
are both separable sums over any partition of the dimensions). cosine
does not decompose that way -- normalization is over the whole vector, not
a sub-vector -- so it's approximated by L2-normalizing vectors before
splitting: on unit vectors, squared L2 distance is a monotonic function of
cosine similarity (||a-b||^2 == 2 - 2*cos_sim(a,b) when ||a||=||b||=1), so
ranking is preserved even though the raw distances aren't literally
1 - cosine_similarity.
"""

from typing import Literal

import numpy as np

from vectordb import distance
from vectordb.index.base import VectorIndex
from vectordb.kmeans import KMeans

Metric = Literal["l2", "cosine", "ip"]

# subspace distance used to build both the codebooks and the Asymmetric Distance Computation (ADC) table
# cosine reuses pairwise_l2 because vectors are normalized before splitting
_SUBSPACE_DIST_FUNCS = {
    "l2": distance.pairwise_l2,
    "cosine": distance.pairwise_l2,
    "ip": distance.pairwise_inner_product,
}


class PQIndex(VectorIndex):

    def __init__(
        self,
        M: int = 8,
        Ks: int = 256,
        metric: Metric = "l2",
    ) -> None:
        if metric not in _SUBSPACE_DIST_FUNCS:
            raise ValueError(
                f"unknown metric {metric!r}, expected one of {sorted(_SUBSPACE_DIST_FUNCS)}"
            )
        if not (1 <= Ks <= 256):
            raise ValueError(f"Ks must be in [1, 256] to fit in a uint8 code, got {Ks}")
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}")
        self.M = M
        self.Ks = Ks
        self.metric = metric

        self.codebooks: np.ndarray | None = None  # (M, Ks, d/M), set by train()
        self.sub_dim: int | None = None  # d / M, set by train()
        self._codes: np.ndarray | None = None  # (N, M) uint8
        self._ids: np.ndarray | None = None

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True) + distance.EPS
        return vectors / norms

    def _split(self, vectors: np.ndarray) -> np.ndarray:
        """Reshape (n, d) into (M, n, d/M) contiguous sub-vector blocks."""
        n, d = vectors.shape
        if d % self.M != 0:
            raise ValueError(f"vector dim {d} not divisible by M={self.M}")
        sub_dim = d // self.M
        return vectors.reshape(n, self.M, sub_dim).transpose(1, 0, 2)

    def train(self, vectors: np.ndarray) -> None:
        """Fit one Ks-centroid k-means per subspace independently; store as (M, Ks, d/M) codebooks."""
        vectors = np.asarray(vectors, dtype=np.float64)
        if self.metric == "cosine":
            vectors = self._normalize(vectors)
        self.sub_dim = vectors.shape[1] // self.M
        sub_vectors = self._split(vectors)  # (M, n, sub_dim)

        codebooks = np.empty((self.M, self.Ks, self.sub_dim)) # type: ignore
        for m in range(self.M):
            km = KMeans(n_clusters=self.Ks, max_iter=200, tol=1e-4)
            km.fit(sub_vectors[m])
            codebooks[m] = np.stack([km.final_centroids[i] for i in range(self.Ks)])
        self.codebooks = codebooks

    def add(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> None:
        """Encode each vector as M nearest-centroid indices (uint8), one per subspace; no full-precision copy kept."""
        assert self.codebooks is not None, "train() must run before add()"
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim == 1:  # cast for shape like (n, d)
            vectors = vectors[None, :]
        if self.metric == "cosine":
            vectors = self._normalize(vectors)
        n = vectors.shape[0]
        if ids is None:
            start = 0 if self._ids is None else int(self._ids[-1]) + 1
            ids = np.arange(start, start + n)
        else:
            ids = np.asarray(ids)
            if ids.shape[0] != n:
                raise ValueError(f"got {n} vectors but {ids.shape[0]} ids")
        # now we encode each vector using the codebook
        # split the vector into subvectors
        sub_vectors = self._split(vectors)  # (M, n, sub_dim)
        dist_func = _SUBSPACE_DIST_FUNCS[self.metric]
        codes = np.empty((n, self.M), dtype=np.uint8)
        for m in range(self.M):
            # for every subvector of segment m, find it's distance to the centroids of that segment
            # sub_vectors[m] is (n, d/m)
            # self.codebooks[m] is (Ks, d/m)
            dists = dist_func(sub_vectors[m], self.codebooks[m])  # (n, Ks)
            # for every subvector, we find which centroid it is closest to
            # so a d dimm vector now gets represented by a M sized code of uint8
            codes[:, m] = np.argmin(dists, axis=1)

        if self._codes is None:
            self._codes = codes
            self._ids = ids
        else:
            self._codes = np.vstack([self._codes, codes])
            self._ids = np.concat((self._ids, ids))  # type: ignore

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """ADC: build an (M, Ks) query-to-centroid distance table, then sum M table lookups per stored code."""
        if self._codes is None or self._codes.shape[0] == 0:
            return np.array([]), np.array([])
        assert self.codebooks is not None, "train() must run before search()"

        query = np.asarray(query, dtype=np.float64).reshape(1, -1)
        if self.metric == "cosine":
            query = self._normalize(query)
        sub_query = self._split(query)  # (M, 1, sub_dim)

        dist_func = _SUBSPACE_DIST_FUNCS[self.metric]
        table = np.empty((self.M, self.Ks))
        # we just compute how far is sub-vector from each segment centroids
        for m in range(self.M):
            table[m] = dist_func(sub_query[m], self.codebooks[m])[0]  # (Ks,)

        n = self._codes.shape[0] # (n, M)
        k = min(k, n)

        # table[m, codes[:, m]] for every m, summed over m
        # approximate distance for every stored vector
        # without ever touching self._codes as anything but a lookup index

        # i.e. now we find the approx distance of sub-query to M-centroid saved vectors
        # d(q, x_i) = sum(j=1...m) T[j][i_j]
        dists = table[np.arange(self.M)[:, None], self._codes.T].sum(axis=0)  # (n,)

        top_k = np.argpartition(dists, k - 1)[:k] # just a fancy way to get topk
        order = np.argsort(dists[top_k])
        top_k = top_k[order]

        return dists[top_k], self._ids[top_k]  # type: ignore
