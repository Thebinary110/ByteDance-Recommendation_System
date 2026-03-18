"""
Semantic vector — maps a movie to a dense relevance vector over the tag vocabulary.

TAG_VOCAB and TAG_DIM are built once at import time from the full MOVIE_TAGS map.
Both are exported so callers can inspect the vocabulary size.
"""

import numpy as np

from tag_service import MOVIE_TAGS, get_tags   # semantic/

# ---------------------------------------------------------------------------
# Vocabulary (built at import time from complete MOVIE_TAGS)
# ---------------------------------------------------------------------------
TAG_VOCAB: dict = {}

def _build_tag_vocab() -> None:
    idx = 0
    for tags in MOVIE_TAGS.values():
        for tag in tags:
            if tag not in TAG_VOCAB:
                TAG_VOCAB[tag] = idx
                idx += 1

_build_tag_vocab()
TAG_DIM: int = len(TAG_VOCAB)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_semantic_vector(movie_id: str) -> np.ndarray:
    """
    Return a float32 vector of length TAG_DIM where each dimension is the
    genome relevance score for the corresponding tag (0.0 if not present).
    """
    vec  = np.zeros(TAG_DIM, dtype=np.float32)
    tags = get_tags(movie_id)
    for tag, score in tags.items():
        idx = TAG_VOCAB.get(tag)
        if idx is not None:
            vec[idx] = float(score)
    return vec
