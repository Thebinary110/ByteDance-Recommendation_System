"""
FAISS index — fast approximate nearest-neighbour retrieval over item embeddings.

Index type: IndexFlatIP (inner product, exact search)
Similarity:  cosine — vectors are L2-normalised before indexing and querying,
             so inner product == cosine similarity.

Rebuild strategy
----------------
Full rebuild every FAISS_REBUILD_INTERVAL events.  Simple and reliable: the
id_map is regenerated from scratch each time so there are no stale entries.
At the scale of MovieLens (~27k items, dim=32) a full rebuild takes < 10ms.

Upgrade path
------------
IndexFlatIP  → exact,  O(N)  per query  (current)
IndexIVFFlat → approx, O(N/n_lists)     (next step when N > 100k)
IndexHNSW    → approx, O(log N)         (production)

Design notes
------------
- id_map[i] gives the item_id string for FAISS vector at position i.
- FAISS returns -1 as a padding index when fewer than top_k results exist;
  the guard `0 <= idx` filters these out (the spec's `idx < len(id_map)`
  alone would incorrectly accept -1).
- The global `faiss_index` instance is imported by inference.py and
  run_embedding.py; both must be in the same Python process to share state.
"""

import faiss
import numpy as np
from typing import List

from emb_config import EMBEDDING_DIM
from embedding_store import item_embeddings


class FaissIndex:
    def __init__(self, dim: int) -> None:
        self.dim   = dim
        self.index = faiss.IndexFlatIP(dim)
        self.id_map: List[str] = []
        self.built  = False

    def build(self) -> int:
        """
        Rebuild the full index from current item_embeddings.
        Returns the number of items indexed.
        """
        if not item_embeddings:
            return 0

        self.id_map = []
        vectors: List[np.ndarray] = []

        for item_id, emb in item_embeddings.items():
            vectors.append(emb.detach().numpy().astype("float32"))
            self.id_map.append(item_id)

        matrix = np.stack(vectors)            # (N, dim)
        faiss.normalize_L2(matrix)            # in-place L2 norm → cosine via IP

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(matrix)
        self.built = True

        return len(self.id_map)

    def search(self, user_vec, top_k: int) -> List[str]:
        """
        Return up to top_k item_ids nearest (by cosine similarity) to user_vec.
        Returns [] if the index has not been built yet.
        """
        if not self.built:
            return []

        vec = user_vec.detach().numpy().astype("float32").reshape(1, -1)
        faiss.normalize_L2(vec)

        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []

        _, indices = self.index.search(vec, k)

        results: List[str] = []
        for idx in indices[0]:
            if 0 <= idx < len(self.id_map):   # guard: FAISS pads with -1
                results.append(self.id_map[idx])

        return results


# Shared global instance — imported by inference.py and run_embedding.py
faiss_index = FaissIndex(dim=EMBEDDING_DIM)
