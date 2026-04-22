"""
Inverse Propensity Scoring (IPS) for debiasing training data.

Motivation: popular items are over-represented in observed ratings because
users are more likely to watch (and hence rate) popular content. Training
on raw ratings therefore amplifies popularity bias. IPS corrects for this
by down-weighting popular items (high propensity) and up-weighting rare
items (low propensity) so the model learns unbiased preferences.

Propensity model: P(item exposed | user) ∝ item_popularity^alpha
  alpha=0 → uniform (no correction)
  alpha=1 → full popularity-proportional propensity
  alpha=0.5 → moderate correction (default, reduces variance vs alpha=1)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_item_popularity(ratings: pd.DataFrame) -> pd.Series:
    """Returns raw interaction count per movie_idx."""
    return ratings.groupby("movie_idx")["rating"].count()


def estimate_propensity(
    ratings: pd.DataFrame,
    num_movies: int,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Estimates exposure probability for each movie.
    Returns array of shape (num_movies,) in [0, 1].
    """
    popularity = compute_item_popularity(ratings)

    # Fill missing movies with count = 1 to avoid zero propensity
    counts = np.ones(num_movies, dtype=np.float64)
    for movie_idx, cnt in popularity.items():
        if 0 <= movie_idx < num_movies:
            counts[movie_idx] = float(cnt)

    # P ∝ count^alpha
    propensity = counts ** alpha
    # Normalise to [0, 1] range (relative propensity)
    propensity = propensity / propensity.max()
    return propensity.astype(np.float32)


def compute_ips_weights(
    ratings: pd.DataFrame,
    num_movies: int,
    alpha: float = 0.5,
    cap: float = 10.0,
) -> np.ndarray:
    """
    Computes per-sample IPS weights for the ratings DataFrame.

    IPS weight = 1 / propensity, capped at `cap` to bound variance.
    Returns array of shape (len(ratings),).
    """
    propensity = estimate_propensity(ratings, num_movies, alpha=alpha)

    movie_idxs = ratings["movie_idx"].values
    sample_propensity = propensity[movie_idxs]

    # Avoid division by zero
    sample_propensity = np.clip(sample_propensity, 1e-6, None)
    weights = 1.0 / sample_propensity

    # Cap to limit variance of the estimator
    weights = np.clip(weights, 1.0, cap)

    # Normalise so mean weight == 1 (keeps loss magnitude stable)
    weights = weights / weights.mean()

    logger.info(
        f"IPS weights — mean: {weights.mean():.3f}, "
        f"max: {weights.max():.3f}, "
        f"min: {weights.min():.3f}"
    )
    return weights.astype(np.float32)


def attach_ips_weights(
    ratings: pd.DataFrame,
    num_movies: int,
    alpha: float = 0.5,
    cap: float = 10.0,
) -> pd.DataFrame:
    """Adds an 'ips_weight' column to the ratings DataFrame in-place."""
    weights = compute_ips_weights(ratings, num_movies, alpha=alpha, cap=cap)
    df = ratings.copy()
    df["ips_weight"] = weights
    return df
