"""
Feature store stream consumer.

Pulls events from the asyncio Queue, passes each to the feature updater,
and logs aggregate stats every LOG_INTERVAL events.
"""

import logging
from asyncio import Queue

from feature_updater import process_event
from fs_config import LOG_INTERVAL
from logger import log_stats

log = logging.getLogger(__name__)


async def consume_events(queue: Queue) -> None:
    count = 0
    log.info("Feature store consumer starting")

    while True:
        event = await queue.get()

        if event is None:
            queue.task_done()
            break

        process_event(event)
        count += 1

        if count % LOG_INTERVAL == 0:
            log_stats(count)

        queue.task_done()

    # Final stats snapshot after stream ends / Ctrl-C
    log_stats(count)
    log.info(f"Feature store consumer done.  total_events={count:,}")
