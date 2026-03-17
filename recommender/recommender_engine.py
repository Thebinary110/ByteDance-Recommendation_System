"""
Recommender engine — orchestrates the full recommendation pipeline for a single user.

Pipeline per request:
  1. Fetch user state from the feature store
  2. Guard: skip if insufficient history
  3. Index user's history into the co-occurrence matrix (once per user)
  4. Generate candidates from liked items via similarity lookup
  5. Filter out items the user has already interacted with
  6. Rank candidates by co-occurrence score
  7. Return top-K

Note on imports:
  user_store is in feature_store/ which must be in sys.path when this module is used.
  run_recommender.py manages sys.path for all three packages.
"""

from typing import List

from candidate_generator import generate_candidates
from ranker import rank_items
from rec_config import MIN_HISTORY, TOP_K
from similarity_engine import update_similarity
from user_store import get_user


def recommend(user_id: str) -> List[str]:
    """Return up to TOP_K recommended item_ids for the given user."""
    user = get_user(user_id)

    if not user:
        return []

    if len(user["history"]) < MIN_HISTORY:
        return []

    # Index this user's history into the co-occurrence matrix.
    # update_similarity is a no-op if already called for this user_id.
    update_similarity(user_id, user["history"])

    candidates = generate_candidates(user)

    if not candidates:
        return []

    # Remove items the user has already seen
    seen = user["likes"] | user["dislikes"]
    candidates = [c for c in candidates if c not in seen]

    if not candidates:
        return []

    ranked = rank_items(user, candidates)
    return ranked[:TOP_K]
