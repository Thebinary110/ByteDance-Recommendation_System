"""
Online evaluation tracker — triggers periodic metric computation during streaming.

Usage (from run_ranker._consume or api/main.py pipeline):
    from online_tracker import track_event

    track_event()   # call once per processed event

Every EVAL_INTERVAL events, evaluate() is called and results are:
  1. Logged at INFO level.
  2. Stored in last_metrics (module-level dict) for the /metrics API endpoint.

last_metrics is intentionally a mutable module-level dict so the API
endpoint can read the most recent results without additional coupling.
"""

import logging

from eval_config import EVAL_INTERVAL, TOP_K
from evaluator import evaluate

log = logging.getLogger("evaluation.tracker")

# Shared state — read by api/routes/evaluation_routes.py
last_metrics: dict = {}

_event_count: int = 0


def track_event() -> None:
    """
    Increment the event counter. Runs evaluate() every EVAL_INTERVAL calls.
    Exceptions inside evaluate() are caught so they never crash the pipeline.
    """
    global _event_count
    _event_count += 1

    if _event_count % EVAL_INTERVAL != 0:
        return

    log.info("Running evaluation at event %d...", _event_count)
    try:
        metrics = evaluate()
        last_metrics.update(metrics)
        log.info(
            "[EVAL] precision@%d=%.4f  recall@%d=%.4f  hit_rate@%d=%.4f  users=%d",
            TOP_K, metrics.get(f"precision@{TOP_K}", 0.0),
            TOP_K, metrics.get(f"recall@{TOP_K}",    0.0),
            TOP_K, metrics.get(f"hit_rate@{TOP_K}",  0.0),
            metrics.get("users_evaluated", 0),
        )
    except Exception as exc:
        log.warning("Evaluation failed at event %d: %s", _event_count, exc)


def get_event_count() -> int:
    """Return total events processed so far."""
    return _event_count
