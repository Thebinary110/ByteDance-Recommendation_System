"""
Standalone evaluation script — runs offline metrics against the live user_store.

Requires the embedding model and feature store to be populated (run the pipeline
first or point to a process that has already loaded data).

Run from the project root:
    python evaluation/run_evaluation.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("feature_store", "embedding_model", "ranking_model", "deepfm", "evaluation"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluator import evaluate
from eval_config import TOP_K


def main() -> None:
    print("Running offline evaluation...")
    metrics = evaluate()

    print("\n--- EVALUATION RESULTS ---")
    print(f"  precision@{TOP_K}  : {metrics[f'precision@{TOP_K}']:.4f}")
    print(f"  recall@{TOP_K}     : {metrics[f'recall@{TOP_K}']:.4f}")
    print(f"  hit_rate@{TOP_K}   : {metrics[f'hit_rate@{TOP_K}']:.4f}")
    print(f"  users evaluated  : {metrics['users_evaluated']:,}")
    print("--------------------------\n")


if __name__ == "__main__":
    main()
