"""
Abstract base class for all vector indexes.

Defines the VectorIndex interface: train → add → search, plus save/load
backed by pickle. Subclasses override train() and add() as needed;
FlatIndex leaves train() as a no-op.

search() always returns (distances, ids) as numpy arrays of shape (k,),
with lower distance meaning closer — consistent across all metrics.
"""

import pickle
from abc import ABC, abstractmethod

import numpy as np


class VectorIndex(ABC):

    @abstractmethod
    def train(self, vectors: np.ndarray) -> None:
        """Fit any index-specific structures (centroids, codebooks, graph) on a sample of vectors. No-op where nothing needs fitting (e.g. FlatIndex)."""
        ...

    @abstractmethod
    def add(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> None:
        """Insert vectors, optionally with caller-supplied ids (default: sequential, continuing from the current size)."""
        ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the k nearest neighbors of query as (distances, ids), both shape (k,), sorted by increasing distance."""
        ...

    def save(self, path: str) -> None:
        """Pickle the whole index, including any trained state, to path."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        """Unpickle an index previously written by save()."""
        with open(path, "rb") as f:
            index = pickle.load(f)
        if not isinstance(index, cls):
            raise TypeError(f"{path!r} does not contain a {cls.__name__}")
        return index
