"""
User service — reads from the feature store and embedding store.

Uses user_embeddings.keys() as the authoritative user list because the
embedding store is populated by every event the ranking pipeline processes.
_store (feature store) is populated by the feature-store consumer and may
lag slightly, so user detail falls back to an empty profile gracefully.
"""

from typing import List, Optional, Dict

# core/config.py must be imported before this module to set up sys.path.
from user_store import _store, get_user          # feature_store
from embedding_store import user_embeddings      # embedding_model


def get_all_users() -> List[str]:
    """Return all user IDs that have embeddings, sorted numerically."""
    ids = list(user_embeddings.keys())
    ids.sort(key=lambda x: int(x) if x.isdigit() else x)
    return ids


def get_user_detail(user_id: str) -> Optional[Dict]:
    """
    Return a JSON-serializable user profile dict, or None if user unknown.
    Converts set → sorted list and deque → len for serialisation safety.
    """
    if user_id not in user_embeddings:
        return None

    user = get_user(user_id)
    if not user:
        # Embedding exists but feature-store consumer hasn't caught up yet
        return {
            "user_id":          user_id,
            "likes":            [],
            "dislikes":         [],
            "history_size":     0,
            "last_interaction": None,
        }

    return {
        "user_id":          user_id,
        "likes":            sorted(user["likes"]),
        "dislikes":         sorted(user["dislikes"]),
        "history_size":     len(user["history"]),
        "last_interaction": user["last_interaction"],
    }
