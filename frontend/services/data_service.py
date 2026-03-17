"""
Data service — all data fetched via HTTP from the FastAPI backend.

The frontend is fully stateless: no backend module imports, no sys.path
manipulation, no in-memory stores. All state lives in the API process.
"""

import time
from typing import Dict, List, Union

import requests

BASE_URL    = "http://localhost:8000"
_TIMEOUT    = 5   # seconds per request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(path: str) -> requests.Response | None:
    """GET with timeout. Returns None on any network/connection error."""
    try:
        return requests.get(f"{BASE_URL}{path}", timeout=_TIMEOUT)
    except requests.exceptions.RequestException:
        return None


# ---------------------------------------------------------------------------
# Public API  (shapes match FastAPI response schemas exactly)
# ---------------------------------------------------------------------------

def get_all_users() -> List[str]:
    """Return sorted list of user IDs. Empty list if API unreachable."""
    r = _get("/users")
    if r and r.status_code == 200:
        return r.json().get("users", [])
    return []


def get_user_data(user_id: str) -> Dict:
    """Return user profile dict (likes, dislikes, history_size, last_interaction)."""
    r = _get(f"/users/{user_id}")
    if r and r.status_code == 200:
        return r.json()
    return {}


def get_recommendations(user_id: str, top_k: int = 10) -> List[Dict]:
    """
    Return enriched recommendation list for user_id.
    Each item: {item_id, title, genres}.
    """
    r = _get(f"/recommend/{user_id}?top_k={top_k}")
    if r and r.status_code == 200:
        return r.json().get("recommendations", [])
    return []


def get_stats() -> Dict:
    """Return system stats (total_users, total_items, faiss_built, …)."""
    r = _get("/stats")
    if r and r.status_code == 200:
        return r.json()
    return {}


def api_latency_ms() -> float:
    """Round-trip latency to GET /stats in milliseconds."""
    t0 = time.perf_counter()
    _get("/stats")
    return round((time.perf_counter() - t0) * 1000, 1)
