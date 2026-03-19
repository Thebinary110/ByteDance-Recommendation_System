"""
Hard negative sampler for Two-Tower training.

Hard negatives are items that are similar to the positive item (high
embedding similarity) but have NOT been interacted with by the user.
They force the model to learn fine-grained distinctions rather than trivially
separating liked items from random noise.

Similarity source:
  tt_item_embeddings (64-dim, same space as training) via dot product.
  This is preferred over FAISS which operates in the 32-dim embedding_model
  space and would create a dimension mismatch.

Fallback:
  When tt_item_embeddings has fewer than MIN_POOL items, fall back to
  random negatives so training never stalls during cold start.

Performance:
  With up to ~27k MovieLens items in tt_item_embeddings, computing dot
  products over the full store takes < 1ms (64-dim float32 vectors).
  A simple per-item cache avoids recomputing for the same positive item
  every event; the cache is invalidated each time the item is retrained.
"""

import logging
from typing import List, Optional, Set

import torch

from tt_store import tt_item_embeddings

log = logging.getLogger("two_tower.hard_neg")

# Minimum items in tt_store before hard negatives kick in
MIN_POOL = 20

# Only flush the similarity cache for an item every N training updates.
# Between flushes the cached list is slightly stale but still much better than
# random negatives — and recomputing per update is the main inference bottleneck.
CACHE_REFRESH_INTERVAL = 10

# Per-item cache: item_id -> sorted list of similar item_ids (most similar first)
_similar_cache:  dict = {}
# Per-item update counter — cache flushed every CACHE_REFRESH_INTERVAL increments
_update_counts:  dict = {}


def invalidate_cache(item_id: str) -> None:
    """
    Called by tt_trainer after retraining item_id.
    Cache is only flushed every CACHE_REFRESH_INTERVAL calls to limit
    the frequency of expensive similarity recomputation.
    """
    count = _update_counts.get(item_id, 0) + 1
    _update_counts[item_id] = count
    if count % CACHE_REFRESH_INTERVAL == 0:
        _similar_cache.pop(item_id, None)


def get_similar_items(pos_item_id: str, top_k: int = 50) -> List[str]:
    """
    Return up to top_k item_ids most similar to pos_item_id by dot product
    over tt_item_embeddings. Excludes pos_item_id itself.

    Results are cached per pos_item_id and invalidated when the item is retrained.
    Returns [] if pos_item_id has no embedding or the store is too small.
    """
    if pos_item_id in _similar_cache:
        return _similar_cache[pos_item_id][:top_k]

    pos_vec = tt_item_embeddings.get(pos_item_id)
    if pos_vec is None or len(tt_item_embeddings) < MIN_POOL:
        return []

    pos_vec = pos_vec.detach()
    scores: List[tuple] = []

    for item_id, item_vec in list(tt_item_embeddings.items()):
        if item_id == pos_item_id:
            continue
        score = torch.dot(pos_vec, item_vec.detach()).item()
        scores.append((item_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    ranked = [item_id for item_id, _ in scores]

    # Cache the full ranking; callers slice to their desired top_k
    _similar_cache[pos_item_id] = ranked
    return ranked[:top_k]


def sample_hard_negatives(
    user_id:     str,
    pos_item_id: str,
    k:           int,
    user_seen:   Optional[Set[str]] = None,
) -> List[str]:
    """
    Return up to k hard negative item_ids for the (user_id, pos_item_id) pair.

    Parameters
    ----------
    user_id     : user whose history is used for filtering
    pos_item_id : the positive item (always excluded from negatives)
    k           : desired number of hard negatives
    user_seen   : pre-computed set of item_ids the user has interacted with.
                  Pass this from tt_trainer to avoid redundant get_user() calls.
                  If None, returns similar items filtered only by pos_item_id.

    Returns
    -------
    List of item_id strings (length <= k). May be shorter than k if the
    similar-item pool is exhausted after filtering; the caller fills the
    remainder with random negatives.
    """
    similar = get_similar_items(pos_item_id, top_k=k * 5)  # wide pool to survive filtering
    if not similar:
        return []

    exclude = {pos_item_id}
    if user_seen:
        exclude |= user_seen

    hard_negs: List[str] = []
    for item_id in similar:
        if item_id not in exclude:
            hard_negs.append(item_id)
        if len(hard_negs) == k:
            break

    return hard_negs
