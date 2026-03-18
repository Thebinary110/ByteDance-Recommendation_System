"""
Link service — O(1) lookup of IMDb / TMDB IDs for any movie.

Loaded once at import time from data/processed/link_map.pkl.
IMDb IDs are stored as 7-digit zero-padded strings (e.g. "0114709").
TMDB IDs are strings or None for the 252 movies with no TMDB entry.
"""

import os
import pickle

_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKL_PATH = os.path.join(_ROOT, "data", "processed", "link_map.pkl")

with open(_PKL_PATH, "rb") as _f:
    LINK_MAP: dict = pickle.load(_f)

_MISSING = {"imdb": None, "tmdb": None}


def get_links(movie_id: str) -> dict:
    """Return {"imdb": "0114709", "tmdb": "862"} or {"imdb": None, "tmdb": None}."""
    return LINK_MAP.get(str(movie_id), _MISSING)
