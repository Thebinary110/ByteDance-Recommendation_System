"""
Feature builder — converts raw user/item/event data into DeepFM-ready tensors.

Genre vocabulary is built once at import time from the full movie map (19 genres).
GENRE_DIM is exposed so deepfm_model.py can read it for its input layer size.
"""

import os
import sys
from typing import Dict, List

import numpy as np

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UTILS_DIR = os.path.join(_ROOT, "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from movie_service import MOVIE_MAP, get_movie  # utils/

# ---------------------------------------------------------------------------
# Genre vocabulary (built once from full movie map)
# ---------------------------------------------------------------------------
GENRE_VOCAB: Dict[str, int] = {}

def _build_vocab() -> None:
    idx = 0
    for movie in MOVIE_MAP.values():
        for g in movie["genres"]:
            if g not in GENRE_VOCAB:
                GENRE_VOCAB[g] = idx
                idx += 1

_build_vocab()
GENRE_DIM: int = len(GENRE_VOCAB)   # 19 for standard MovieLens


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_genres(genres: List[str]) -> np.ndarray:
    """Return a multi-hot float32 vector of length GENRE_DIM."""
    vec = np.zeros(GENRE_DIM, dtype=np.float32)
    for g in genres:
        if g in GENRE_VOCAB:
            vec[GENRE_VOCAB[g]] = 1.0
    return vec


def safe_int_id(raw_id: str, modulus: int) -> int:
    """Convert string ID to int, clamped to [0, modulus). Safe for any input."""
    try:
        return int(raw_id) % modulus
    except (ValueError, TypeError):
        return abs(hash(raw_id)) % modulus


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def build_features(
    user_id: str,
    item_id: str,
    user_emb,           # torch.Tensor (already detached)
    item_emb,           # torch.Tensor (already detached)
) -> Dict:
    """
    Return a dict of model-ready numpy arrays and int IDs.
    user_emb / item_emb are expected to be detached before this call.
    """
    from deepfm_model import NUM_USERS, NUM_ITEMS   # avoid circular at module level

    movie     = get_movie(item_id)
    genre_vec = encode_genres(movie["genres"])

    return {
        "user_id_int": safe_int_id(user_id, NUM_USERS),
        "item_id_int": safe_int_id(item_id, NUM_ITEMS),
        "user_emb":    user_emb.numpy().copy(),
        "item_emb":    item_emb.numpy().copy(),
        "genre_vec":   genre_vec,
    }
