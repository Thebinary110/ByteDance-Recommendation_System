"""
TMDB poster service — fetches and caches poster URLs for movies.

API key is read from the TMDB_API_KEY environment variable.
Returns None gracefully when: key is missing, movie not found, network error.

In-memory cache (_poster_cache) avoids redundant API calls for the same
movie across multiple requests in the same server process.
"""

import logging
import os
from typing import Optional

import requests

log = logging.getLogger("tmdb")

_API_KEY    = os.environ.get("TMDB_API_KEY", "")
_BASE_URL   = "https://api.themoviedb.org/3/movie"
_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"   # w342 = good balance of size/quality
_TIMEOUT    = 4   # seconds — tight so a slow TMDB doesn't stall the API response

# Module-level cache: tmdb_id (str) → poster URL or None
_poster_cache: dict = {}


def get_poster(tmdb_id: Optional[str]) -> Optional[str]:
    """
    Return the full poster image URL for tmdb_id, or None if unavailable.

    Results are cached in-process so repeated calls for the same movie
    are free after the first fetch.
    """
    if not tmdb_id:
        return None
    if not _API_KEY:
        log.debug("TMDB_API_KEY not set — skipping poster fetch")
        return None

    if tmdb_id in _poster_cache:
        return _poster_cache[tmdb_id]

    url = f"{_BASE_URL}/{tmdb_id}?api_key={_API_KEY}"
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        if resp.status_code == 200:
            path = resp.json().get("poster_path")
            result = (_IMAGE_BASE + path) if path else None
        else:
            result = None
    except requests.exceptions.RequestException as exc:
        log.debug("TMDB fetch failed for %s: %s", tmdb_id, exc)
        result = None

    _poster_cache[tmdb_id] = result
    return result
