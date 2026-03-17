"""
Producer — reads events.csv, validates each event, and fans out to all consumer queues.

Replay mode: sleeps proportional to the timestamp gap between consecutive events,
compressed by SCALE_FACTOR and capped at MAX_REPLAY_DELAY so large jumps don't hang.

Fixed mode: sleeps STREAM_DELAY between every event.
"""

import asyncio
import csv
import logging
import os
import time
from typing import Optional

from config import (
    EVENTS_PATH,
    LOG_INTERVAL,
    MAX_REPLAY_DELAY,
    REPLAY_MODE,
    SCALE_FACTOR,
    STREAM_DELAY,
)
from queue_manager import get_all_queues

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"user_id", "item_id", "event_type", "timestamp"}


def validate_event(event: dict) -> None:
    """Raise ValueError if any required field is missing."""
    missing = REQUIRED_FIELDS - event.keys()
    if missing:
        raise ValueError(f"Missing required fields: {missing}")


async def produce() -> None:
    if not os.path.exists(EVENTS_PATH):
        logger.error(f"Events file not found: {EVENTS_PATH}")
        logger.error("Run the data pipeline first: python data_pipeline/run_pipeline.py")
        for q in get_all_queues():
            await q.put(None)
        return

    count = 0
    skipped = 0
    t0 = time.perf_counter()
    prev_ts: Optional[int] = None

    logger.info(
        f"Producer starting — source: {EVENTS_PATH}  "
        f"replay={REPLAY_MODE}  consumers={len(get_all_queues())}"
    )

    with open(EVENTS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:

            # 1. Schema validation
            try:
                validate_event(row)
            except ValueError as exc:
                logger.warning(f"[PRODUCER] Skipping invalid event — {exc}  row={row}")
                skipped += 1
                continue

            # 2. Timing
            if REPLAY_MODE:
                current_ts = int(row["timestamp"])
                if prev_ts is not None:
                    time_diff = current_ts - prev_ts
                    if time_diff > 0:
                        sleep_time = min(time_diff / SCALE_FACTOR, MAX_REPLAY_DELAY)
                        await asyncio.sleep(sleep_time)
                prev_ts = current_ts
            else:
                await asyncio.sleep(STREAM_DELAY)

            # 3. Fan-out to all consumer queues
            for q in get_all_queues():
                await q.put(row)

            count += 1

            if count % LOG_INTERVAL == 0:
                elapsed = time.perf_counter() - t0
                rate = count / elapsed
                logger.info(
                    f"[PRODUCER] {count:>10,} pushed  |  "
                    f"{rate:,.0f} events/s  |  consumers={len(get_all_queues())}"
                )

    # Sentinel: signal every consumer to stop
    for q in get_all_queues():
        await q.put(None)

    elapsed = time.perf_counter() - t0
    logger.info(
        f"[PRODUCER] Done.  total={count:,}  skipped={skipped}  "
        f"elapsed={elapsed:.1f}s  avg={count / elapsed:,.0f} events/s"
    )
