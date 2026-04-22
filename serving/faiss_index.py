"""
FAISS-based Approximate Nearest Neighbour index for candidate retrieval.

Uses IndexFlatIP (exact inner product) for the ~27K movie catalog,
which fits comfortably in memory and gives exact results.
For catalogs >1M, swap to IndexIVFFlat with appropriate nlist.

After L2-normalisation in TwoTowerModel, inner product == cosine similarity.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning(
        "faiss-cpu not installed. Using brute-force numpy search as fallback. "
        "Install with: pip install faiss-cpu"
    )

logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    Wraps a FAISS inner-product index with a consistent interface.
    Falls back to exact numpy brute-force search when faiss isn't installed.
    """

    def __init__(self):
        self._index = None          # faiss.Index or None
        self._embeddings: np.ndarray | None = None  # (num_items, dim) fallback
        self.num_items: int = 0
        self.embed_dim: int = 0
        self._use_faiss = FAISS_AVAILABLE

    def build(self, embeddings: np.ndarray) -> "FAISSIndex":
        """
        Build the index from a (num_items, dim) float32 array.
        Embeddings are assumed L2-normalised (unit norm).
        """
        assert embeddings.ndim == 2, "embeddings must be 2-D"
        self.num_items, self.embed_dim = embeddings.shape
        vecs = embeddings.astype(np.float32, copy=False)

        if self._use_faiss:
            self._index = faiss.IndexFlatIP(self.embed_dim)
            self._index.add(vecs)
            logger.info(
                f"FAISS IndexFlatIP built: {self._index.ntotal} vectors, dim={self.embed_dim}"
            )
        else:
            # Fallback: store embeddings and use numpy matmul
            self._embeddings = vecs
            logger.info(
                f"Numpy brute-force index built: {self.num_items} vectors, dim={self.embed_dim}"
            )
        return self

    def search(
        self, query: np.ndarray, k: int = 200
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k items closest to query.

        Parameters
        ----------
        query : (dim,) or (n_queries, dim) float32 array
        k     : number of results per query

        Returns
        -------
        scores  : (n_queries, k) float32
        indices : (n_queries, k) int64
        """
        query = np.atleast_2d(query).astype(np.float32)
        k = min(k, self.num_items)

        if self._use_faiss and self._index is not None:
            scores, indices = self._index.search(query, k)
        elif self._embeddings is not None:
            # Brute-force: inner product = dot product (vecs are unit-normed)
            sims = query @ self._embeddings.T  # (n_q, num_items)
            top_k_idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            top_k_scores = np.take_along_axis(sims, top_k_idx, axis=1)
            # Sort within the top-k block
            order = np.argsort(-top_k_scores, axis=1)
            indices = np.take_along_axis(top_k_idx, order, axis=1)
            scores = np.take_along_axis(top_k_scores, order, axis=1)
        else:
            raise RuntimeError("Index not built. Call build() first.")

        return scores, indices

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._use_faiss and self._index is not None:
            faiss.write_index(self._index, str(path))
            logger.info(f"FAISS index saved → {path}")
        elif self._embeddings is not None:
            np.save(str(path) + ".npy", self._embeddings)
            logger.info(f"Numpy fallback index saved → {path}.npy")

    def load(self, path: str | Path) -> "FAISSIndex":
        path = Path(path)
        if self._use_faiss and path.exists():
            self._index = faiss.read_index(str(path))
            self.num_items = self._index.ntotal
            self.embed_dim = self._index.d
            logger.info(
                f"FAISS index loaded: {self.num_items} vectors, dim={self.embed_dim}"
            )
        else:
            npy_path = Path(str(path) + ".npy")
            if npy_path.exists():
                self._embeddings = np.load(str(npy_path))
                self.num_items, self.embed_dim = self._embeddings.shape
                self._use_faiss = False
                logger.info(
                    f"Numpy fallback index loaded: {self.num_items} vectors"
                )
            else:
                raise FileNotFoundError(f"No index found at {path} or {npy_path}")
        return self
