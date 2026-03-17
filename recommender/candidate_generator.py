"""
Candidate generator — expands the user's liked items into a set of unseen candidates.

For each liked item, retrieves the top-N most similar items from the co-occurrence
index and unions them into a candidate pool.
"""

from typing import List

from rec_config import SIMILARITY_TOP_N
from similarity_engine import get_similar_items


def generate_candidates(user: dict) -> List[str]:
    """
    Return a deduplicated list of candidate item_ids sourced from
    items similar to those the user has liked.
    """
    candidates: set = set()

    for item_id in user["likes"]:
        similar_items = get_similar_items(item_id, SIMILARITY_TOP_N)
        for sim_item, _ in similar_items:
            candidates.add(sim_item)

    return list(candidates)
