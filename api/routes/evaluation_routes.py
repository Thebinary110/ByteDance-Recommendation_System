"""
GET /metrics — returns the most recent evaluation results.

Results are populated by online_tracker.track_event() every EVAL_INTERVAL
streaming events. Returns a status message before the first evaluation runs.
"""

from fastapi import APIRouter

from online_tracker import get_event_count, last_metrics

router = APIRouter(tags=["evaluation"])


@router.get("/metrics")
def get_metrics() -> dict:
    """
    Returns the latest offline evaluation metrics.

    Fields (after first evaluation run):
        precision@10, recall@10, hit_rate@10, users_evaluated

    Before the first run:
        {"status": "no evaluation yet", "events_processed": N}
    """
    if not last_metrics:
        return {
            "status":           "no evaluation yet",
            "events_processed": get_event_count(),
        }
    return dict(last_metrics)
