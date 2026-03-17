"""
Structured logger for feature store stats.

Outputs two lines every LOG_INTERVAL events:
  1. Aggregate stats  — event count, user count, avg history size, total interactions
  2. Sample user state — first user's likes, dislikes, recent history, last interaction
"""

import logging

from user_store import get_sample_user, get_stats

log = logging.getLogger("feature_store.stats")


def log_stats(count: int) -> None:
    stats = get_stats()
    log.info(
        f"events={count:>10,}  "
        f"users={stats['total_users']:>7,}  "
        f"avg_history={stats['avg_history_size']:>5.1f}  "
        f"total_interactions={stats['total_interactions']:>10,}"
    )

    sample = get_sample_user()
    if sample:
        user_id, state = sample
        log.info(
            f"sample user_id={user_id!r}  "
            f"likes={state['likes']}  "
            f"dislikes={state['dislikes']}  "
            f"history(last5)={state['history']}  "
            f"last_interaction={state['last_interaction']}"
        )
