"""
Recommender system orchestrator — runs the full pipeline end-to-end.

Flow:
  streaming producer → asyncio Queue → feature store consumer (inline)
                                              ↓  every FEATURE_LOG_INTERVAL events
                                        log_stats()
                                              ↓  every REC_SAMPLE_INTERVAL events
                                        recommend() for TEST_USERS → logged

Run from the project root:
    python recommender/run_recommender.py

Import path strategy
--------------------
Three sibling packages all share sys.path.  Name collisions are avoided by using
distinct config file names in each package:
  streaming/   → config.py      imported as "config"
  feature_store/ → fs_config.py  imported as "fs_config"
  recommender/ → rec_config.py  imported as "rec_config"

sys.path order:  [recommender/, feature_store/, streaming/, ...]
  - recommender modules (rec_config, similarity_engine, ...) resolve via recommender/
  - feature_store modules (fs_config, user_store, ...)        resolve via feature_store/
  - streaming modules (config, queue_manager, producer)       resolve via streaming/
  - "consumer" → feature_store/consumer.py wins over streaming/consumer.py (correct)
"""

import asyncio
import logging
import os
import sys
from asyncio import Queue

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REC_DIR = os.path.dirname(os.path.abspath(__file__))
_FS_DIR = os.path.join(_ROOT, "feature_store")
_STREAMING_DIR = os.path.join(_ROOT, "streaming")

sys.path.insert(0, _STREAMING_DIR)
sys.path.insert(0, _FS_DIR)
sys.path.insert(0, _REC_DIR)

from feature_updater import process_event        # feature_store/feature_updater.py
from logger import log_stats                     # feature_store/logger.py
from producer import produce                     # streaming/producer.py
from queue_manager import register_consumer      # streaming/queue_manager.py
from recommender_engine import recommend         # recommender/recommender_engine.py

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# --- Configuration ---
TEST_USERS = ["1", "2", "3", "10"]     # user_ids sampled for recommendation display
FEATURE_LOG_INTERVAL = 10_000           # feature store stats every N events
REC_SAMPLE_INTERVAL = 50_000            # recommendation snapshot every N events


def _show_recommendations(event_count: int) -> None:
    logger.info(f"--- Recommendations @ {event_count:,} events ---")
    for user_id in TEST_USERS:
        recs = recommend(user_id)
        logger.info(f"  user={user_id!r}  →  {recs if recs else '(not enough data yet)'}")


async def _consume(queue: Queue) -> None:
    """Inline consumer: updates feature store and samples recommendations periodically."""
    count = 0

    while True:
        event = await queue.get()

        if event is None:
            queue.task_done()
            break

        process_event(event)
        count += 1

        if count % FEATURE_LOG_INTERVAL == 0:
            log_stats(count)

        if count % REC_SAMPLE_INTERVAL == 0:
            _show_recommendations(count)

        queue.task_done()

    log_stats(count)
    _show_recommendations(count)


async def main() -> None:
    logger.info("Recommender system starting")
    logger.info(f"Test users: {TEST_USERS}")
    logger.info(f"Recommendations shown every {REC_SAMPLE_INTERVAL:,} events")

    queue = register_consumer("feature_store")
    await asyncio.gather(produce(), _consume(queue))
    logger.info("Recommender system complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[STOPPED] Recommender interrupted.")
