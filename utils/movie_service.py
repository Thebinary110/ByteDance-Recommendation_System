"""
Movie metadata service — loaded once at import time.

Provides O(1) lookup from item_id (str) to {title, genres}.
Falls back gracefully for unknown IDs.
"""

import os
import pickle
from typing import Dict, List

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MAP_PATH  = os.path.join(_ROOT, "data", "processed", "movie_map.pkl")

with open(_MAP_PATH, "rb") as _f:
    MOVIE_MAP: Dict[str, Dict] = pickle.load(_f)


def get_movie(movie_id: str) -> Dict:
    """Return {title, genres} for movie_id, or a fallback dict if unknown."""
    return MOVIE_MAP.get(str(movie_id), {
        "title":  f"Unknown ({movie_id})",
        "genres": [],
    })


def get_title(movie_id: str) -> str:
    return get_movie(movie_id)["title"]


def get_genres(movie_id: str) -> List[str]:
    return get_movie(movie_id)["genres"]
