"""
Explainability engine — answers "why was this item recommended?"

For each (user, recommended_item) pair it produces:
  reason_items  — titles of the user's liked items most similar to the recommendation
  reason_scores — cosine similarity expressed as a percentage (bonus)
  shared_genres — genres the recommendation shares with the user's liked items (bonus)

Performance notes:
  - Cosine similarity is computed on 32-dim float32 vectors (~880 ops for a user
    with 88 likes × 10 recommendations — negligible latency).
  - .detach().numpy().copy() is used so the numpy array is a stable snapshot
    that is unaffected if the training thread modifies the tensor's .data in-place.
  - item_embeddings.get(key) is a single dict lookup (no iteration), so no
    "changed size during iteration" risk.
  - liked_genres is computed once per call and reused across genre overlap check.
"""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

# utils/ (project root) must be on sys.path for movie_service
_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_UTILS_DIR = os.path.join(_ROOT, "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from embedding_store import item_embeddings  # embedding_model
from user_store import get_user              # feature_store
from movie_service import get_movie          # utils


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
    similar = get_top_similar_liked_items(user_id, item_id, top_n=2)

    reason_items:  List[str]   = []
    reason_scores: List[float] = []
    for liked_id, score in similar:
        reason_items.append(get_movie(liked_id)["title"])
        reason_scores.append(round(score * 100, 1))

    shared_genres = get_shared_genres(item_id, user_id, liked_genre_cache)

    return {
        "reason_items":  reason_items,
        "reason_scores": reason_scores,
        "shared_genres": shared_genres,
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
