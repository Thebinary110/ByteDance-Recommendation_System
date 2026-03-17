"""
Stream orchestrator — registers consumers, then starts producer + all consumers concurrently.

Run from the project root:
    python streaming/run_stream.py

Add or remove consumer names from CONSUMERS to scale fan-out.
"""

import asyncio
import logging
import os
import sys

# Ensure sibling modules resolve when invoked from the project root.
sys.path.insert(0, os.path.dirname(__file__))

from consumer import consume
from producer import produce
from queue_manager import register_consumer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# Add or remove names here to scale consumers.
CONSUMERS = ["model", "logger", "feature_store"]


async def main() -> None:
    logger.info(f"Stream initializing — consumers: {CONSUMERS}")

    consumer_coroutines = [
        consume(name, register_consumer(name))
        for name in CONSUMERS
    ]

    await asyncio.gather(produce(), *consumer_coroutines)
    logger.info("Stream complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[STOPPED] Stream interrupted by user.")
