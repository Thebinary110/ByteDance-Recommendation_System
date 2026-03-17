"""
Item-to-item co-occurrence similarity engine.

Co-occurrence definition: two items are similar if they appear together
in the same user's interaction history. The more users who interacted
with both, the stronger the similarity.

Update strategy:
  Each user's history is indexed exactly once (tracked via _indexed_users).
  Calling update_similarity() for the same user repeatedly is a no-op,
  preventing count inflation if recommend() is called multiple times.
"""

from collections import defaultdict
from typing import List, Tuple

# item_cooccurrence[item_a][item_b] = number of users who interacted with both
item_cooccurrence: defaultdict = defaultdict(lambda: defaultdict(int))

# Tracks which users have already been indexed to prevent double-counting
_indexed_users: set = set()


def update_similarity(user_id: str, user_history) -> None:
    """
    Index a user's interaction history into the co-occurrence matrix.
    Each user_id is processed only once — subsequent calls are no-ops.
    """
    if user_id in _indexed_users:
        return

    items = [item_id for item_id, _, _ in user_history]

    for item_i in items:
        for item_j in items:
            if item_i != item_j:
                item_cooccurrence[item_i][item_j] += 1

    _indexed_users.add(user_id)


def get_similar_items(item_id: str, top_n: int) -> List[Tuple[str, int]]:
    """Return the top_n most co-occurring items for item_id, sorted by count descending."""
    similar = item_cooccurrence.get(item_id, {})
    return sorted(similar.items(), key=lambda x: x[1], reverse=True)[:top_n]


def get_cooccurrence_score(item_a: str, item_b: str) -> int:
    """Return the co-occurrence count between two items (0 if no data)."""
    return item_cooccurrence[item_a].get(item_b, 0)
