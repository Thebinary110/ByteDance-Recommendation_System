"""
In-memory user state store.

Each user entry holds:
  - likes      : set of item_ids the user has liked
  - dislikes   : set of item_ids the user has disliked
  - history    : bounded deque of (item_id, event_type, timestamp) tuples
  - last_interaction : timestamp of the most recent event

Design notes:
  - deque(maxlen=MAX_HISTORY) enforces bounded memory automatically.
  - likes/dislikes are mutually exclusive: liking an item removes it from dislikes and vice versa.
  - RLock is used for future thread-safety (e.g. thread pool executors, Redis migration).
    In the current single-threaded asyncio runtime it is never contended.
"""

import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from fs_config import MAX_HISTORY

_lock = threading.RLock()
_store: Dict[str, Dict] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_user(user_id: str) -> None:
    """Create a fresh entry for user_id if one does not exist. Caller holds _lock."""
    if user_id not in _store:
        _store[user_id] = {
            "likes": set(),
            "dislikes": set(),
            "history": deque(maxlen=MAX_HISTORY),
            "last_interaction": None,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_user(user_id: str) -> Optional[Dict]:
    """Return the raw state dict for user_id, or None if the user has no events yet."""
    with _lock:
        return _store.get(user_id)


def update_user(user_id: str, item_id: str, event_type: str, timestamp: str) -> None:
    """Apply a single event to the user's state."""
    with _lock:
        _ensure_user(user_id)
        user = _store[user_id]

        if event_type == "like":
            user["likes"].add(item_id)
            user["dislikes"].discard(item_id)   # an item can't be both
        elif event_type == "dislike":
            user["dislikes"].add(item_id)
            user["likes"].discard(item_id)

        user["history"].append((item_id, event_type, int(timestamp)))
        user["last_interaction"] = int(timestamp)


def get_stats() -> Dict[str, Any]:
    """Return aggregate statistics across all users."""
    with _lock:
        total_users = len(_store)
        if total_users == 0:
            return {"total_users": 0, "avg_history_size": 0.0, "total_interactions": 0}
        total_interactions = sum(len(u["history"]) for u in _store.values())
        return {
            "total_users": total_users,
            "avg_history_size": round(total_interactions / total_users, 1),
            "total_interactions": total_interactions,
        }


def get_sample_user() -> Optional[Tuple[str, Dict]]:
    """
    Return (user_id, summarized_state) for the first user in the store.
    Caps output to 5 items per field so log lines stay readable.
    """
    with _lock:
        if not _store:
            return None
        user_id, user = next(iter(_store.items()))
        return user_id, {
            "likes":    sorted(user["likes"])[:5],
            "dislikes": sorted(user["dislikes"])[:5],
            "history":  list(user["history"])[-5:],
            "last_interaction": user["last_interaction"],
        }
