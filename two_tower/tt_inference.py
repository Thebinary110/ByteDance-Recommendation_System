"""
Two-Tower retrieval — dot-product scoring over trained item embeddings.

Primary path (after training has started):
  1. Look up cached user vector from tt_user_embeddings.
  2. Dot-product score against all cached tt_item_embeddings (L2-normalised,
     so dot product = cosine similarity).
  3. Filter excluded items, return top_k.

Fallback path (cold start — no tt embeddings yet):
  Delegates to embedding_model/inference.py (FAISS / linear scan on 32-dim
  vectors), so the system is never stuck with empty recommendations.

Thread-safety note:
  list(tt_item_embeddings.items()) takes a snapshot before scoring, preventing
  "dict changed size during iteration" if the training coroutine runs concurrently.
"""

import logging
from typing import List, Optional, Set

import torch

from tt_store import tt_item_embeddings, tt_user_embeddings

log = logging.getLogger("two_tower.inference")


def retrieve(
    user_id: str,
    top_k:   int = 50,
    exclude: Optional[Set[str]] = None,
) -> List[str]:
    """
    Return up to top_k item_ids ranked by two-tower dot-product similarity.
    Falls back to FAISS retrieval during cold start (no tt embeddings).
    """
    _exclude = exclude or set()

    # Cold-start fallback — two-tower not trained yet
    if user_id not in tt_user_embeddings or not tt_item_embeddings:
        log.debug("tt cold start for user %s — delegating to FAISS", user_id)
        try:
            from inference import recommend as faiss_retrieve  # embedding_model
            return faiss_retrieve(user_id, top_k=top_k, exclude=_exclude)
        except Exception as exc:
            log.warning("FAISS fallback failed: %s", exc)
            return []

    user_vec = tt_user_embeddings[user_id].detach()

    scores: List[tuple] = []
    for item_id, item_vec in list(tt_item_embeddings.items()):
        if item_id in _exclude:
            continue
        score = torch.dot(user_vec, item_vec.detach()).item()
        scores.append((item_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in scores[:top_k]]
