"""
Recommendation service — two-stage pipeline with explainability + metadata.

Pipeline:
  1. FAISS top-50 retrieval (embedding similarity), excluding seen items.
  2. DeepFM reranking -> top-k.
  3. Explanation layer: per-item reason (similar liked items) + shared genres
     + semantic reason (genome tag vectors).
  4. Metadata enrichment: TMDB poster URL + IMDb link.

The liked_genre_cache (union of all liked items' genres) is built once per
request and passed into each build_explanation call to avoid redundant lookups.
TMDB poster URLs are cached in tmdb_service after the first fetch.
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
from tt_inference import retrieve            # two_tower        (learned retrieval + FAISS fallback)
from deepfm_inference import rank_items      # deepfm           (DeepFM reranker)
from movie_service import get_movie          # utils
from link_service import get_links           # utils
from tmdb_service import get_poster          # utils
from services.explanation_service import build_explanation, build_liked_genre_cache


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
        links       = get_links(item_id)
        poster      = get_poster(links["tmdb"])
        imdb_url    = (
            f"https://www.imdb.com/title/tt{links['imdb']}"
            if links["imdb"] else None
        )
        enriched.append({
            "item_id":         item_id,
            "title":           movie["title"],
            "genres":          movie["genres"],
            "reason":          explanation["reason_items"],
            "reason_scores":   explanation["reason_scores"],
            "shared_genres":   explanation["shared_genres"],
            "semantic_reason": explanation["semantic_reason"],
            "poster":          poster,
            "imdb_url":        imdb_url,
        })

    return enriched
