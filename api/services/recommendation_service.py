"""
Recommendation service — two-stage pipeline with explainability.

Pipeline:
  1. FAISS top-50 retrieval (embedding similarity), excluding seen items.
  2. MLP reranking -> top-k.
  3. Explanation layer: per-item reason (similar liked items) + shared genres.

The liked_genre_cache (union of all liked items' genres) is built once per
request and passed into each build_explanation call to avoid repeated lookups.
"""

import os
import sys
from typing import Dict, List

# utils/ for movie_service and explanation_service
_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_UTILS_DIR = os.path.join(_ROOT, "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from user_store import get_user              # feature_store
from embedding_store import user_embeddings  # embedding_model
from inference import recommend as retrieve  # embedding_model  (FAISS / linear)
from rank_inference import rank_items        # ranking_model    (MLP reranker)
from movie_service import get_movie          # utils
from explanation_service import build_explanation, build_liked_genre_cache  # api/services


def get_recommendations(user_id: str, top_k: int = 10) -> List[Dict]:
    """
    Return up to top_k enriched + explained recommendation dicts.
    Each dict: {item_id, title, genres, reason, reason_scores, shared_genres}.
    Returns [] if the user has no embedding yet.
    """
    if user_id not in user_embeddings:
        return []

    user    = get_user(user_id)
    exclude = (user["likes"] | user["dislikes"]) if user else set()

    candidates = retrieve(user_id, top_k=50, exclude=exclude)
    ranked     = rank_items(user_id, candidates, top_k=top_k)

    # Pre-compute once — reused across all items in the loop
    liked_genre_cache = build_liked_genre_cache(user_id)

    enriched = []
    for item_id in ranked:
        movie       = get_movie(item_id)
        explanation = build_explanation(user_id, item_id, liked_genre_cache)
        enriched.append({
            "item_id":       item_id,
            "title":         movie["title"],
            "genres":        movie["genres"],
            "reason":        explanation["reason_items"],
            "reason_scores": explanation["reason_scores"],
            "shared_genres": explanation["shared_genres"],
        })

    return enriched
