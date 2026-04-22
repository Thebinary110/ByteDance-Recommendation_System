"""
User Simulator for offline evaluation of the recommendation policy.

Why simulate users?
  Offline evaluation typically measures precision/recall against held-out ratings,
  but this misses the sequential, contextual nature of real interactions.
  The simulator models a user as a bandit arm whose behaviour follows a
  logistic function of item relevance, enabling us to measure:
    • Cumulative reward (clicks / ratings) over a session
    • Exploration–exploitation trade-off
    • Effects of re-ranking on engagement

Simulator model:
  P(click | user, item) = σ( true_pref(user, item) + context_noise )
  where true_pref = user_features ⊙ item_features (dot product)
  and context_noise ~ N(0, noise_std²)

  P(high_rating | click) = σ( true_pref - rating_threshold )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SimulatedSession:
    user_id: int
    clicks: list[int] = field(default_factory=list)
    skips: list[int] = field(default_factory=list)
    ratings: dict[int, float] = field(default_factory=dict)
    total_reward: float = 0.0


class UserSimulator:
    """
    Simulates user interactions given a recommendation list.

    Parameters
    ----------
    user_features  : (num_users, feat_dim) array of user feature vectors
    item_features  : (num_movies, feat_dim) array of item feature vectors
    rating_df      : ground-truth ratings DataFrame (user_idx, movie_idx, rating)
                     used to calibrate the simulator's preference model
    noise_std      : standard deviation of per-interaction Gaussian noise
    rating_threshold : logit threshold above which a click becomes a high-rating
    """

    def __init__(
        self,
        user_features: np.ndarray,
        item_features: np.ndarray,
        noise_std: float = 0.5,
        rating_threshold: float = 0.2,
    ):
        self.user_features = user_features.astype(np.float32)
        self.item_features = item_features.astype(np.float32)
        self.noise_std = noise_std
        self.rating_threshold = rating_threshold
        self.rng = np.random.default_rng(42)

    def _true_preference(self, user_idx: int, movie_idx: int) -> float:
        """Dot product preference score (unnormalised)."""
        u = self.user_features[user_idx]
        v = self.item_features[movie_idx]
        # Use only as many dims as both share
        d = min(len(u), len(v))
        return float(np.dot(u[:d], v[:d]))

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def simulate_click(self, user_idx: int, movie_idx: int) -> bool:
        """Stochastically decide if the user clicks on the item."""
        pref = self._true_preference(user_idx, movie_idx)
        noise = self.rng.normal(0, self.noise_std)
        p_click = self._sigmoid(pref + noise)
        return self.rng.random() < p_click

    def simulate_rating(self, user_idx: int, movie_idx: int) -> float:
        """Sample a rating given that the user clicked (in [1, 5])."""
        pref = self._true_preference(user_idx, movie_idx)
        noise = self.rng.normal(0, self.noise_std * 0.5)
        logit = pref + noise - self.rating_threshold
        p_high = self._sigmoid(logit)
        # Map probability to a 1–5 rating
        raw = 1.0 + 4.0 * p_high + self.rng.normal(0, 0.3)
        return float(np.clip(raw, 1.0, 5.0))

    def run_session(
        self,
        user_idx: int,
        recommended_ids: list[int],
        position_bias: bool = True,
    ) -> SimulatedSession:
        """
        Simulate a full session for one user.

        position_bias: items at rank r get click probability multiplied by 1/log2(r+2),
                       modelling the fact that users rarely scroll past the first few rows.
        """
        session = SimulatedSession(user_id=user_idx)
        for rank, movie_idx in enumerate(recommended_ids):
            click = self.simulate_click(user_idx, movie_idx)
            if position_bias:
                decay = 1.0 / np.log2(rank + 2)
                click = click and (self.rng.random() < decay)

            if click:
                session.clicks.append(movie_idx)
                rating = self.simulate_rating(user_idx, movie_idx)
                session.ratings[movie_idx] = rating
                # Reward = normalised rating
                session.total_reward += (rating - 1.0) / 4.0
            else:
                session.skips.append(movie_idx)

        return session

    def evaluate_policy(
        self,
        recommender_fn: Any,  # callable(user_idx) -> list[int] of movie_idxs
        test_user_idxs: list[int],
        top_k: int = 10,
    ) -> dict[str, float]:
        """
        Evaluate a recommendation policy over a set of test users.

        recommender_fn should accept a user_idx (int) and return a list of movie_idx.
        Returns aggregated metrics.
        """
        ctr_list, reward_list, cov_items = [], [], set()

        for user_idx in test_user_idxs:
            recs = recommender_fn(user_idx)[:top_k]
            if not recs:
                continue
            session = self.run_session(user_idx, recs, position_bias=True)

            ctr = len(session.clicks) / max(len(recs), 1)
            ctr_list.append(ctr)
            reward_list.append(session.total_reward)
            cov_items.update(session.clicks)

        return {
            "simulated_ctr": float(np.mean(ctr_list)) if ctr_list else 0.0,
            "simulated_avg_reward": float(np.mean(reward_list)) if reward_list else 0.0,
            "simulated_coverage": len(cov_items),
            "n_users_evaluated": len(ctr_list),
        }
