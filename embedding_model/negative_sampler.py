"""
Negative sampler — draws random items the user has not interacted with.

Design notes:
  - Maintains its own lightweight per-user seen-item set (_user_seen) so the
    embedding model stays fully standalone and does not depend on the feature store.
  - record_interaction() must be called before sample_negatives() for each event
    so the current item is already excluded from the negative pool.
  - Uses random.choice() in a bounded loop (spec pattern) with a max_attempts
    guard to prevent infinite loops when almost all items are seen.
  - Sampling from list(item_embeddings.keys()) ensures we only draw items that
    already have embeddings — no cold-start negatives with zero-signal tensors.
"""

import random
from typing import List

from embedding_store import item_embeddings

# Per-user set of interacted item_ids — kept minimal (only IDs, no vectors)
_user_seen: dict = {}


def record_interaction(user_id: str, item_id: str) -> None:
    """Mark item_id as seen by user_id so it is excluded from future negatives."""
    if user_id not in _user_seen:
        _user_seen[user_id] = set()
    _user_seen[user_id].add(item_id)


def sample_negatives(user_id: str, num_samples: int) -> List[str]:
    """
    Return up to num_samples item_ids that user_id has not yet interacted with.

    May return fewer than num_samples if the item pool is too small or dense
    with seen items (bounded by max_attempts to keep latency O(1) in practice).
    """
    seen      = _user_seen.get(user_id, set())
    all_items = list(item_embeddings.keys())

    if not all_items:
        return []

    negatives: List[str] = []
    seen_negatives: set  = set()
    max_attempts         = num_samples * 20   # prevents slow paths when seen ≈ all_items

    attempts = 0
    while len(negatives) < num_samples and attempts < max_attempts:
        item = random.choice(all_items)
        if item not in seen and item not in seen_negatives:
            negatives.append(item)
            seen_negatives.add(item)
        attempts += 1

    return negatives
