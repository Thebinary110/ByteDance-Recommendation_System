"""
Offline evaluation for the full recommendation pipeline.

Metrics computed:
  • NDCG@k   — measures ranking quality, rewards correct items ranked high
  • Precision@k — fraction of recommended items that are relevant
  • Recall@k  — fraction of relevant items that appear in top-k
  • MRR       — mean reciprocal rank of first relevant item
  • Coverage  — fraction of catalog recommended at least once
  • ILD       — Intra-List Diversity (from reranker.py)
  • Simulated CTR / reward — from UserSimulator

Usage:
  python -m training.evaluate --output_dir artifacts/ --top_k 10 --n_users 500
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from data.preprocessor import Preprocessor
from models.reranker import intra_list_diversity
from models.user_simulator import UserSimulator

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Ranking metrics
# ------------------------------------------------------------------

def _dcg(relevances: list[float], k: int) -> float:
    gains = [r / np.log2(i + 2) for i, r in enumerate(relevances[:k])]
    return sum(gains)


def ndcg_at_k(predicted: list[int], relevant: set[int], k: int = 10) -> float:
    rel = [1.0 if p in relevant else 0.0 for p in predicted[:k]]
    ideal = sorted(rel, reverse=True)
    dcg = _dcg(rel, k)
    idcg = _dcg(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(predicted: list[int], relevant: set[int], k: int = 10) -> float:
    hits = sum(1 for p in predicted[:k] if p in relevant)
    return hits / k


def recall_at_k(predicted: list[int], relevant: set[int], k: int = 10) -> float:
    hits = sum(1 for p in predicted[:k] if p in relevant)
    return hits / max(len(relevant), 1)


def mrr(predicted: list[int], relevant: set[int]) -> float:
    for rank, p in enumerate(predicted, start=1):
        if p in relevant:
            return 1.0 / rank
    return 0.0


# ------------------------------------------------------------------
# Full evaluation
# ------------------------------------------------------------------

def evaluate_full(
    recommender_fn: Callable[[int], list[int]],
    test_data,    # encoded test DataFrame
    prep: Preprocessor,
    user_simulator: UserSimulator | None = None,
    top_k: int = 10,
    n_users: int = 500,
    item_embeddings: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Evaluate a recommender over n_users sampled from test_data.

    recommender_fn(user_idx) → list of movie_idx recommendations
    """
    # Build ground-truth relevant items per user from test data
    relevant_items: dict[int, set[int]] = (
        test_data[test_data["rating"] >= 3.5]
        .groupby("user_idx")["movie_idx"]
        .apply(set)
        .to_dict()
    )

    test_users = list(relevant_items.keys())
    if len(test_users) > n_users:
        test_users = random.sample(test_users, n_users)

    metrics_per_user: list[dict] = []
    covered_items: set[int] = set()

    for user_idx in test_users:
        relevant = relevant_items.get(user_idx, set())
        if not relevant:
            continue

        recs = recommender_fn(user_idx)[:top_k]
        if not recs:
            continue

        covered_items.update(recs)
        m = {
            "ndcg": ndcg_at_k(recs, relevant, top_k),
            "precision": precision_at_k(recs, relevant, top_k),
            "recall": recall_at_k(recs, relevant, top_k),
            "mrr": mrr(recs, relevant),
        }
        if item_embeddings is not None:
            m["ild"] = intra_list_diversity(recs, item_embeddings)
        metrics_per_user.append(m)

    results: dict[str, Any] = {}
    for key in ["ndcg", "precision", "recall", "mrr", "ild"]:
        vals = [m[key] for m in metrics_per_user if key in m]
        if vals:
            results[f"{key}@{top_k}"] = float(np.mean(vals))

    results["coverage"] = len(covered_items) / prep.num_movies
    results["n_users_evaluated"] = len(metrics_per_user)

    # Simulator metrics
    if user_simulator is not None:
        sim_metrics = user_simulator.evaluate_policy(
            recommender_fn, test_users[:min(200, len(test_users))], top_k
        )
        results.update(sim_metrics)

    return results


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    output_dir = Path(args.output_dir)

    prep = Preprocessor.load(output_dir)
    with open(output_dir / "splits.pkl", "rb") as f:
        splits = pickle.load(f)
    test_data = splits["test"]

    # Load item embeddings for ILD
    item_embeddings = None
    emb_path = output_dir / "item_embeddings.npy"
    if emb_path.exists():
        item_embeddings = np.load(emb_path)

    # Import the serving engine
    from serving.inference import RecommendationEngine
    engine = RecommendationEngine.load(output_dir)

    def rec_fn(user_idx: int) -> list[int]:
        try:
            return engine.recommend_by_idx(user_idx, limit=args.top_k)
        except Exception:
            return []

    # User simulator
    simulator = None
    if args.simulate:
        simulator = UserSimulator(
            prep.user_features, prep.item_features, noise_std=0.5
        )

    results = evaluate_full(
        rec_fn,
        test_data,
        prep,
        user_simulator=simulator,
        top_k=args.top_k,
        n_users=args.n_users,
        item_embeddings=item_embeddings,
    )

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k:<30}: {v:.4f}" if isinstance(v, float) else f"  {k:<30}: {v}")

    out_path = output_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="artifacts")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--n_users", type=int, default=500)
    parser.add_argument("--simulate", action="store_true")
    main(parser.parse_args())
