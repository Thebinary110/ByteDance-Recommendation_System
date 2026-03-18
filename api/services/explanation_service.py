"""
Explainability engine — answers "why was this item recommended?"

For each (user, recommended_item) pair it produces:
  reason_items   — titles of the user's liked items most similar (embedding cosine)
  reason_scores  — cosine similarity expressed as a percentage
  shared_genres  — genres the recommendation shares with the user's liked items
  semantic_reason — titles of user's liked items most conceptually similar
                    (MovieLens genome tag vectors)

Performance notes:
  - Embedding cosine similarity: 32-dim float32 vectors, negligible latency.
  - Semantic cosine similarity: TAG_DIM-dim vectors; computed on demand, cached
    per session via Python's import-time module state in semantic_vector.py.
  - .detach().numpy().copy() avoids data-race with training thread's in-place updates.
  - liked_genre_cache computed once per request, reused across all 10 items.
"""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

# utils/ and semantic/ must be on sys.path
_ROOT         = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_UTILS_DIR    = os.path.join(_ROOT, "utils")
_SEMANTIC_DIR = os.path.join(_ROOT, "semantic")
for _p in (_UTILS_DIR, _SEMANTIC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from embedding_store import item_embeddings          # embedding_model
from user_store import get_user                      # feature_store
from movie_service import get_movie                  # utils
from semantic_similarity import semantic_similarity  # semantic


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]."""
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / denom)


def _safe_emb(item_id: str):
    """Return a stable numpy copy of item_id's embedding, or None if missing."""
    tensor = item_embeddings.get(item_id)
    if tensor is None:
        return None
    return tensor.detach().numpy().copy()


# ---------------------------------------------------------------------------
# Core explanation functions
# ---------------------------------------------------------------------------

def get_top_similar_liked_items(
    user_id: str,
    item_id: str,
    top_n: int = 2,
) -> List[Tuple[str, float]]:
    """
    Return top_n (liked_item_id, similarity_score) pairs whose embeddings are
    most similar to the target item's embedding.
    """
    user = get_user(user_id)
    if not user or not user["likes"]:
        return []

    target_emb = _safe_emb(item_id)
    if target_emb is None:
        return []

    sims: List[Tuple[str, float]] = []
    for liked_id in user["likes"]:
        liked_emb = _safe_emb(liked_id)
        if liked_emb is None:
            continue
        sims.append((liked_id, _cosine_sim(target_emb, liked_emb)))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_n]


def get_shared_genres(
    item_id: str,
    user_id: str,
    liked_genre_cache: set = None,
) -> List[str]:
    """
    Return genres the recommended item shares with the user's liked items.
    Pass liked_genre_cache to avoid recomputing per-user genre union on every call.
    """
    target_genres = get_movie(item_id)["genres"]
    if not target_genres:
        return []

    if liked_genre_cache is None:
        user = get_user(user_id)
        liked_genre_cache = set()
        if user:
            for liked_id in user["likes"]:
                liked_genre_cache.update(get_movie(liked_id)["genres"])

    shared = set(target_genres) & liked_genre_cache
    return [g for g in target_genres if g in shared]   # preserve genre order


def get_top_semantic_items(
    user_id: str,
    item_id: str,
    top_n: int = 2,
) -> List[Tuple[str, float]]:
    """
    Return top_n (liked_item_id, semantic_similarity) pairs whose genome tag
    vectors are most similar to the target item's tag vector.
    Returns [] if user has no likes or item has no genome data.
    """
    user = get_user(user_id)
    if not user or not user["likes"]:
        return []

    sims: List[Tuple[str, float]] = []
    for liked_id in user["likes"]:
        score = semantic_similarity(item_id, liked_id)
        if score > 0.0:
            sims.append((liked_id, score))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_n]


def build_explanation(
    user_id: str,
    item_id: str,
    liked_genre_cache: set = None,
) -> Dict:
    """
    Build a full explanation dict for one (user, item) pair.
    Pass liked_genre_cache (pre-computed per-user genre union) to avoid
    redundant lookups when calling this inside a loop over ranked items.
    """
    similar  = get_top_similar_liked_items(user_id, item_id, top_n=2)
    semantic = get_top_semantic_items(user_id, item_id, top_n=2)

    reason_items:  List[str]   = []
    reason_scores: List[float] = []
    for liked_id, score in similar:
        reason_items.append(get_movie(liked_id)["title"])
        reason_scores.append(round(score * 100, 1))

    semantic_reason: List[str] = [get_movie(i)["title"] for i, _ in semantic]
    shared_genres = get_shared_genres(item_id, user_id, liked_genre_cache)

    return {
        "reason_items":   reason_items,
        "reason_scores":  reason_scores,
        "shared_genres":  shared_genres,
        "semantic_reason": semantic_reason,
    }


def build_liked_genre_cache(user_id: str) -> set:
    """Pre-compute the union of all genres from a user's liked items."""
    user = get_user(user_id)
    if not user or not user["likes"]:
        return set()
    cache: set = set()
    for liked_id in user["likes"]:
        cache.update(get_movie(liked_id)["genres"])
    return cache
