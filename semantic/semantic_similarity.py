"""
Semantic similarity — cosine similarity between two movies' genome tag vectors.

Returns 0.0 for movies absent from the genome data (zero vectors).
"""

import numpy as np

from semantic_vector import get_semantic_vector   # semantic/


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1] (genome vectors are non-negative)."""
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / denom)


def semantic_similarity(movie_a: str, movie_b: str) -> float:
    """
    Cosine similarity between movie_a and movie_b based on genome tag vectors.
    Returns 0.0 if either movie is missing from the genome data.
    """
    vec_a = get_semantic_vector(movie_a)
    vec_b = get_semantic_vector(movie_b)
    return cosine_sim(vec_a, vec_b)
