"""
Consumer — pulls events from its dedicated queue and processes them.

Each consumer is identified by a name (e.g. "model", "logger", "feature_store").
All logging goes through the named logger so output is attributable per consumer.

Will be extended to feed into model / feature store in later stages.
"""

import logging
import time
from asyncio import Queue

from config import LOG_INTERVAL

logger = logging.getLogger(__name__)


async def consume(name: str, queue: Queue) -> None:
    consumer_log = logging.getLogger(f"consumer.{name}")
    count = 0
    t0 = time.perf_counter()

    consumer_log.info(f"[{name}] Consumer starting")

    while True:
        event = await queue.get()

        if event is None:
            queue.task_done()
            break

        count += 1

        if count % LOG_INTERVAL == 0:
            elapsed = time.perf_counter() - t0
            rate = count / elapsed
            consumer_log.info(
                f"[{name}] {count:>10,}  "
                f"user={event['user_id']:<8}  "
                f"item={event['item_id']:<8}  "
                f"type={event['event_type']:<8}  "
                f"ts={event['timestamp']}  |  "
                f"{rate:,.0f} events/s  queue={queue.qsize()}"
            )

        queue.task_done()

    elapsed = time.perf_counter() - t0
    consumer_log.info(
        f"[{name}] Done.  total={count:,}  "
        f"elapsed={elapsed:.1f}s  avg={count / elapsed:,.0f} events/s"
    )
