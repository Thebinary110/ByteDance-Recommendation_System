"""
Evaluator — computes precision@K, recall@K, hit_rate@K across all eligible users.

Design notes:
  - Calls FAISS retrieve() then DeepFM rank_items() directly (not the API service)
    to avoid test-leakage: the API service excludes ALL of a user's likes from
    recommendations, which would include test items and make metrics always 0.
    Here we exclude only train_items so test items can be retrieved and scored.
  - Falls back to MLP rank_inference if DeepFM is unavailable.
  - Users without embeddings are skipped (not yet seen by the embedding model).
"""

import logging
import os
import sys

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("embedding_model", "ranking_model", "deepfm", "feature_store"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

log = logging.getLogger("evaluation.evaluator")

from embedding_store import user_embeddings            # embedding_model
from inference import recommend as retrieve            # embedding_model (FAISS)

# Prefer DeepFM ranker; fall back to MLP if not available
try:
    from deepfm_inference import rank_items
    log.debug("Using DeepFM ranker for evaluation")
except ImportError:
    from rank_inference import rank_items              # ranking_model (MLP)
    log.debug("Using MLP ranker for evaluation (DeepFM not found)")

from dataset_builder import build_eval_dataset
from metrics import hit_rate_at_k, precision_at_k, recall_at_k

from eval_config import TOP_K, TOP_K_CANDIDATES


def evaluate(
    min_interactions: int   = None,
    test_ratio:       float = None,
) -> dict:
    """
    Run offline evaluation against the current live user_store.

    Parameters
    ----------
    min_interactions : override eval_config default for dataset building
    test_ratio       : override eval_config default for dataset building

    Returns a dict:
        {
            "precision@K": float,
            "recall@K":    float,
            "hit_rate@K":  float,
            "users_evaluated": int,
        }
    """
    dataset = build_eval_dataset(
        min_interactions=min_interactions,
        test_ratio=test_ratio,
    )
    if not dataset:
        return {
            f"precision@{TOP_K}": 0.0,
            f"recall@{TOP_K}":    0.0,
            f"hit_rate@{TOP_K}":  0.0,
            "users_evaluated":    0,
        }

    p_sum = r_sum = h_sum = 0.0
    n_users = 0

    for user_id, splits in dataset.items():
        if user_id not in user_embeddings:
            continue  # embedding not yet trained for this user

        train_set = set(splits["train"])
        test_set  = set(splits["test"])

        # Retrieve excluding only train items — test items CAN appear in results
        try:
            candidates = retrieve(user_id, top_k=TOP_K_CANDIDATES, exclude=train_set)
            ranked     = rank_items(user_id, candidates, top_k=TOP_K)
        except Exception as exc:
            log.debug("Inference failed for user %s: %s", user_id, exc)
            continue

        p_sum += precision_at_k(ranked, test_set, TOP_K)
        r_sum += recall_at_k(ranked, test_set, TOP_K)
        h_sum += hit_rate_at_k(ranked, test_set, TOP_K)
        n_users += 1

    if n_users == 0:
        return {
            f"precision@{TOP_K}": 0.0,
            f"recall@{TOP_K}":    0.0,
            f"hit_rate@{TOP_K}":  0.0,
            "users_evaluated":    0,
        }

    return {
        f"precision@{TOP_K}": round(p_sum / n_users, 4),
        f"recall@{TOP_K}":    round(r_sum / n_users, 4),
        f"hit_rate@{TOP_K}":  round(h_sum / n_users, 4),
        "users_evaluated":    n_users,
    }
