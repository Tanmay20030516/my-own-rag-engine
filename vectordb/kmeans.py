"""
Lloyd's k-means clustering implemented in numpy.

Used internally by IVFIndex (to build inverted lists) and PQIndex (to train
per-subspace codebooks). Not part of the public index API.

KMeans(n_clusters, max_iter, tol).fit(X) returns (centroids, labels).
"""

from typing import Literal

import numpy as np

from vectordb import distance


class KMeans:

    def __init__(
        self,
        n_clusters: int,
        max_iter: int,
        tol: float,
        init_method: Literal["kmeans++", "random"] = "kmeans++",
    ) -> None:
        """Store clustering hyperparameters; no data touched until fit()."""
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method

    def _init_centroids(self):
        """Pick starting centroids via kmeans++ (distance-weighted) or uniform random."""
        if self.init_method == "kmeans++":
            # --- original (unvectorized) implementation, kept for reference ---
            # centroidIdx = [np.random.randint(self.num_samples)]
            # for l in range(self.n_clusters - 1):
            #     scores = []
            #     for i in range(self.num_samples):
            #         score_i = float("inf")
            #         for ci in centroidIdx:
            #             score_i = min(
            #                 score_i, np.sum((self.X_train[i] - self.X_train[ci]) ** 2)
            #             )
            #         scores.append(score_i)
            #     scores = np.array(scores) / sum(scores)
            #     nextCentroidIdx = np.random.choice(self.num_samples, p=scores)
            #     centroidIdx.append(nextCentroidIdx)
            # return self.X_train[centroidIdx]

            # vectorized: keep a running "distance to nearest chosen centroid"
            # array and update it with one broadcasted call per new centroid,
            # instead of recomputing distances to every previous centroid from
            # scratch each round (O(n*k) total instead of O(n*k^2))
            centroidIdx = [np.random.randint(self.num_samples)]
            min_dists = distance.l2_squared(self.X_train, self.X_train[centroidIdx[0]])
            for _ in range(self.n_clusters - 1):
                scores = min_dists / min_dists.sum()
                nextCentroidIdx = np.random.choice(self.num_samples, p=scores)
                centroidIdx.append(nextCentroidIdx)
                new_dists = distance.l2_squared(self.X_train, self.X_train[nextCentroidIdx])
                min_dists = np.minimum(min_dists, new_dists)
            return self.X_train[centroidIdx]

        elif self.init_method == "random":
            indices = np.random.randint(self.num_samples, size=self.n_clusters)
            return self.X_train[indices]
            # return indices
        else:
            print(
                f"Uknown centroid initialization method: {self.init_method}; returning random centroids"
            )
            indices = np.random.randint(self.num_samples, size=self.n_clusters)
            return self.X_train[indices]

    def _centroid_matrix(self, centroids_t: dict) -> np.ndarray:
        """
        Stack a {cluster_id: centroid} dict into a (n_clusters, d) matrix,
        ordered by cluster id 0..n_clusters-1.

        Safe because every centroids dict passed around this class (from
        _init_centroids via fit(), or from _compute_mean) is always fully
        populated over range(n_clusters) -- _compute_mean explicitly fills
        or reinitializes any empty cluster before returning.
        """
        return np.stack([centroids_t[k] for k in range(self.n_clusters)])

    def _assign_centroids(self, centroids_t):
        """Expectation step"""
        # --- original (unvectorized) implementation, kept for reference ---
        # assignments_tp1 = []
        # for i in range(self.num_samples):
        #     best_centroid = None
        #     best_distance = float("inf")
        #     for k, centroid in centroids_t.items():
        #         curr_distance = np.sum((self.X_train[i] - centroid) ** 2)
        #         if curr_distance < best_distance:
        #             best_distance = curr_distance
        #             best_centroid = k
        #     assignments_tp1.append(best_centroid)
        # return np.array(assignments_tp1)

        # vectorized: one (num_samples, n_clusters) distance matrix via
        # distance.pairwise_l2 instead of num_samples * n_clusters scalar
        # np.sum() calls in a Python double loop
        centroid_matrix = self._centroid_matrix(centroids_t)
        dists = distance.pairwise_l2(self.X_train, centroid_matrix)  # (num_samples, n_clusters)
        return np.argmin(dists, axis=1)

    def _compute_mean(self, assignments_t):
        """Maximization step"""
        point_idx = {k: [] for k in range(self.n_clusters)}
        for i, z_t in enumerate(assignments_t):
            point_idx[z_t].append(i)

        centroids_t = {}
        for k, idxs in point_idx.items():
            if idxs:
                centroids_t[k] = np.mean(self.X_train[idxs], axis=0)

        empty_clusters = [k for k in range(self.n_clusters) if k not in centroids_t]
        if empty_clusters:
            # Reinitialize each empty cluster to the point currently farthest
            # from its own centroid, stealing it from a non-empty cluster.
            # Reassigned locally so the next stolen point can't be reused.
            assignments_t = np.array(assignments_t, copy=True)
            for k in empty_clusters:
                worst_idx, worst_dist = None, -1.0
                for j, z_t in enumerate(assignments_t):
                    if z_t in empty_clusters:
                        # edge-case:
                        # to avoid a stolen point be chosen again
                        # i.e. a point was stolen for k1; now won't be chosen for k2
                        # k1, k2 belong to empty_clusters
                        continue
                    dist = np.sum((self.X_train[j] - centroids_t[z_t]) ** 2)
                    if dist > worst_dist:
                        worst_dist = dist
                        worst_idx = j
                centroids_t[k] = self.X_train[worst_idx].copy()
                assignments_t[worst_idx] = k

        return centroids_t

    def _cost(self, assignments, centroids):
        """Mean squared distance from each point to its assigned centroid."""
        # --- original (unvectorized) implementation, kept for reference ---
        # cost = 0.0
        # for i, z_t in enumerate(assignments):
        #     cost += np.sum((self.X_train[i] - centroids[z_t]) ** 2)
        # return cost / self.num_samples

        # vectorized: fancy-index each point's assigned centroid in one shot,
        # then a single elementwise diff + sum instead of a per-point loop
        centroid_matrix = self._centroid_matrix(centroids)
        assigned_centroids = centroid_matrix[assignments]
        cost = np.sum((self.X_train - assigned_centroids) ** 2)
        return cost / self.num_samples

    def fit(self, X: np.ndarray):
        """Run Lloyd's algorithm on X until cost improvement drops below tol or max_iter is hit."""
        self.X_train = X
        self.num_samples, self.num_feat = X.shape
        prev_cost = float("inf")
        centroids_t = {k: v for k, v in enumerate(self._init_centroids())}
        assignments_tp1, centroids_tp1 = None, centroids_t
        for it in range(self.max_iter):
            assignments_tp1 = self._assign_centroids(centroids_t)
            centroids_tp1 = self._compute_mean(assignments_tp1)

            curr_cost = self._cost(assignments_tp1, centroids_tp1)

            if abs(prev_cost - curr_cost) < self.tol:
                break

            prev_cost = curr_cost
            centroids_t = centroids_tp1

        self.final_assignments = assignments_tp1
        self.final_centroids = centroids_tp1

    def predict(self, X):
        """Assign each row of X to its nearest centroid from a fitted model."""
        # --- original (unvectorized) implementation, kept for reference ---
        # n, _ = X.shape
        # predictions = []
        # for i in range(n):
        #     closest_centroid = None
        #     dist = float("inf")
        #     for k, centroid in self.final_centroids.items():
        #         curr_dist = np.sum((X[i] - centroid) ** 2)
        #         if curr_dist < dist:
        #             dist = curr_dist
        #             closest_centroid = k
        #     predictions.append(closest_centroid)
        # return np.array(predictions)

        # vectorized: same distance-matrix + argmin trick as _assign_centroids,
        # instead of a per-row Python loop
        centroid_matrix = self._centroid_matrix(self.final_centroids)
        dists = distance.pairwise_l2(X, centroid_matrix)
        return np.argmin(dists, axis=1)
