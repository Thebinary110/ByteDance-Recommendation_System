"""
Ranker — scores and orders candidate items for a given user.

Scoring heuristic:
  score(candidate) = Σ cooccurrence(liked_item, candidate)  for all liked_item in user["likes"]

Rationale: a candidate that co-occurred many times with many of the user's liked items
is more relevant than one that only co-occurred with a single liked item.
"""

from typing import List

from similarity_engine import get_cooccurrence_score


def rank_items(user: dict, candidates: List[str]) -> List[str]:
    """Return candidates sorted by descending co-occurrence score with the user's liked items."""
    liked = user["likes"]

    scores = {
        candidate: sum(get_cooccurrence_score(liked_item, candidate) for liked_item in liked)
        for candidate in candidates
    }

    return sorted(scores, key=scores.get, reverse=True)
