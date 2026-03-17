"""
Inference — item retrieval via FAISS nearest-neighbour search.

Primary path (after first FAISS build):
  1. Normalize user embedding.
  2. Query FAISS index for top_k * 2 candidates (buffer for exclude filtering).
  3. Filter out already-seen items.
  4. Return top_k.

Fallback path (before first FAISS build):
  Linear dot-product scan over all item embeddings — same as Stage 5.
  Active only for the first FAISS_REBUILD_INTERVAL events (~20k) then
  never used again unless the index is explicitly reset.

Why top_k * 2 for the FAISS query:
  The exclude filter removes already-interacted items.  Fetching 2× gives
  enough buffer so the final list is rarely shorter than top_k.
  For users with very large seen sets (> top_k items), fewer results may
  be returned — acceptable for this baseline.
"""

import torch
from typing import List, Set

from embedding_store import item_embeddings, user_embeddings
from faiss_index import faiss_index


def recommend(user_id: str, top_k: int = 10, exclude: Set[str] = None) -> List[str]:
    """
    Return up to top_k item_ids for user_id.
    Uses FAISS if the index has been built, otherwise falls back to linear scan.
    """
    if user_id not in user_embeddings:
        return []

    user_emb = user_embeddings[user_id]
    _exclude = exclude or set()

    # --- FAISS path ---
    if faiss_index.built:
        candidates = faiss_index.search(user_emb, top_k * 2)
        if _exclude:
            candidates = [c for c in candidates if c not in _exclude]
        return candidates[:top_k]

    # --- Linear fallback (pre-build) ---
    if not item_embeddings:
        return []

    scores: List[tuple] = []
    with torch.no_grad():
        u = user_emb.detach()
        for item_id, item_emb in item_embeddings.items():
            if item_id in _exclude:
                continue
            scores.append((item_id, torch.dot(u, item_emb.detach()).item()))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [iid for iid, _ in scores[:top_k]]
