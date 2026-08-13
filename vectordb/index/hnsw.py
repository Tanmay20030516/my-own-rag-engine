"""
HNSWIndex — hierarchical navigable small world graph ANN index.

Builds a multi-layer proximity graph (Malkov & Yashunin, see resources.txt):
layer 0 holds every vector, each higher layer holds an exponentially
thinner random subset. add() picks a random top layer for the new node,
greedily descends from the current entry point down to that layer, then
at each layer from there to 0 runs a beam search for nearby nodes and
connects the new node to its M nearest, pruning any node whose degree
grows past its cap. search() does the same greedy descent through the
sparse upper layers to find a good entry point, then a single wider beam
search in the dense layer 0.

No training step -- the graph is built incrementally as vectors are added.
"""

import heapq
import math
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


class HNSWIndex(VectorIndex):

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        metric: Metric = "l2",
    ) -> None:
        if metric not in _PAIRWISE_FUNCS:
            raise ValueError(
                f"unknown metric {metric!r}, expected one of {sorted(_PAIRWISE_FUNCS)}"
            )
        if M < 2:
            # ln(M) is the level-assignment denominator below; M=1 makes it 0
            raise ValueError(f"M must be >= 2, got {M}")
        self.M = M
        self.M_max0 = 2 * M  # layer 0 is kept denser than higher layers -- standard HNSW choice
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.metric = metric
        self._level_mult = 1.0 / math.log(M)

        self.layers: list[dict[int, list[int]]] = []  # layers[l][node_id] -> neighbor node_ids at level l
        self.node_level: dict[int, int] = {}  # node_id -> top level it was inserted at
        self.entry_point: int | None = None
        self.top_level: int = -1

        self._vectors: np.ndarray | None = None
        self._ids: np.ndarray | None = None

    def train(self, vectors: np.ndarray) -> None:
        """No-op: the graph is built incrementally by add(), nothing to fit upfront."""

    def _dist(self, q: np.ndarray, idxs) -> np.ndarray:
        """Distance from a single query row (1, d) to self._vectors[idxs], as a 1-D array."""
        return _PAIRWISE_FUNCS[self.metric](q, self._vectors[idxs])[0]

    def _random_level(self) -> int:
        """Exponential-decay level assignment: most nodes land at layer 0, few reach high layers."""
        return int(math.floor(-math.log(np.random.uniform()) * self._level_mult))

    def _search_layer(
        self, q: np.ndarray, entry_points: list[int], ef: int, layer: int
    ) -> list[tuple[float, int]]:
        """
        Beam search a single layer starting from entry_points.

        Standard HNSW beam search: `candidates` is a min-heap of unexplored
        nodes to expand, `found` is a max-heap (negated) capped at ef holding
        the best results seen so far. Expansion stops once the nearest
        unexplored candidate is farther than the current worst of `found`.
        Returns up to ef (distance, node_id) pairs sorted by increasing
        distance.
        """
        visited = set(entry_points)
        entry_dists = self._dist(q, list(entry_points))
        candidates = list(zip((float(d) for d in entry_dists), entry_points))
        heapq.heapify(candidates)
        found = [(-d, n) for d, n in candidates]
        heapq.heapify(found)

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            worst_found = -found[0][0]
            if dist_c > worst_found and len(found) >= ef:
                break

            neighbors = [n for n in self.layers[layer].get(c, []) if n not in visited]
            if not neighbors:
                continue
            visited.update(neighbors)

            n_dists = self._dist(q, neighbors)
            for d, n in zip(n_dists, neighbors):
                d = float(d)
                worst_found = -found[0][0]
                if len(found) < ef or d < worst_found:
                    heapq.heappush(candidates, (d, n))
                    heapq.heappush(found, (-d, n))
                    if len(found) > ef:
                        heapq.heappop(found)

        return sorted(((-d, n) for d, n in found), key=lambda pair: pair[0])

    def _connect(self, node_id: int, neighbor_id: int, layer: int) -> None:
        """
        Add a bidirectional edge at `layer`, pruning either endpoint back to
        its degree cap if needed.

        Pruning only trims the busy node's own list -- it doesn't remove the
        matching entry from whichever neighbor got dropped. That's standard
        HNSW behavior (same in hnswlib's shrink-connections step), not a
        bug: over many inserts the graph can end up with one-directional
        edges, and search still works because traversal follows whichever
        side still lists the edge.
        """
        max_degree = self.M_max0 if layer == 0 else self.M
        for a, b in ((node_id, neighbor_id), (neighbor_id, node_id)):
            neighbors = self.layers[layer].setdefault(a, [])
            if b not in neighbors:
                neighbors.append(b)
            if len(neighbors) > max_degree:
                dists = self._dist(self._vectors[a : a + 1], neighbors)
                keep = np.argsort(dists)[:max_degree]
                self.layers[layer][a] = [neighbors[i] for i in keep]

    def _insert_one(self, row_idx: int) -> None:
        q = self._vectors[row_idx : row_idx + 1]
        level = self._random_level()

        for _ in range(len(self.layers), level + 1):
            self.layers.append({})
        self.node_level[row_idx] = level

        if self.entry_point is None:
            self.entry_point = row_idx
            self.top_level = level
            return

        # phase 1: greedily descend through layers above the new node's
        # level, one nearest-node hop at a time, to find a good starting
        # point for the wider search below
        ep_set = [self.entry_point]
        for lc in range(self.top_level, level, -1):
            ep_set = [self._search_layer(q, ep_set, ef=1, layer=lc)[0][1]]

        # phase 2: beam search + connect at every layer from min(top, level) to 0
        for lc in range(min(self.top_level, level), -1, -1):
            candidates = self._search_layer(q, ep_set, ef=self.ef_construction, layer=lc)
            neighbors = candidates[: self.M]
            for _, n in neighbors:
                self._connect(row_idx, n, lc)
            ep_set = [n for _, n in candidates] or ep_set

        if level > self.top_level:
            self.entry_point = row_idx
            self.top_level = level

    def add(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> None:
        """Append vectors (and ids, default sequential), inserting each one into the graph in turn."""
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim == 1:  # cast for shape like (n, d)
            vectors = vectors[None, :]
        n = vectors.shape[0]

        if ids is None:
            start = 0 if self._ids is None else int(self._ids[-1]) + 1
            ids = np.arange(start, start + n)
        else:
            ids = np.asarray(ids)
            if ids.shape[0] != n:
                raise ValueError(f"got {n} vectors but {ids.shape[0]} ids")

        row_start = 0 if self._vectors is None else self._vectors.shape[0]
        if self._vectors is None:
            self._vectors = vectors
            self._ids = ids
        else:
            self._vectors = np.vstack([self._vectors, vectors])
            self._ids = np.concat((self._ids, ids))  # type: ignore

        for offset in range(n):
            self._insert_one(row_start + offset)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Greedy descent to layer 1 for an entry point, then one ef_search-wide beam search at layer 0."""
        if self.entry_point is None:
            return np.array([]), np.array([])

        query = np.asarray(query, dtype=np.float64).reshape(1, -1)

        ep_set = [self.entry_point]
        for lc in range(self.top_level, 0, -1):
            ep_set = [self._search_layer(query, ep_set, ef=1, layer=lc)[0][1]]

        ef = max(self.ef_search, k)
        found = self._search_layer(query, ep_set, ef=ef, layer=0)[:k]
        if not found:
            return np.array([]), np.array([])

        dists = np.array([d for d, _ in found])
        idxs = np.array([n for _, n in found], dtype=int)
        return dists, self._ids[idxs]  # type: ignore
