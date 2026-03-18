"""
Tag service — loads movie_tags.pkl once at import time.

Provides:
    get_tags(movie_id)  -> dict {tag: relevance}   (empty dict if unknown)
    MOVIE_TAGS          -> full mapping for vocab construction
"""

import os
import pickle

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKL_PATH  = os.path.join(_ROOT, "data", "processed", "movie_tags.pkl")

with open(_PKL_PATH, "rb") as _f:
    MOVIE_TAGS: dict = pickle.load(_f)


def get_tags(movie_id: str) -> dict:
    """Return {tag: relevance} for movie_id, or {} if not in genome data."""
    return MOVIE_TAGS.get(str(movie_id), {})
