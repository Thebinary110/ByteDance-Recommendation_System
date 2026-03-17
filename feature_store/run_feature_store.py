"""
Feature store orchestrator — wires the streaming producer to the feature store consumer.

Run from the project root:
    python feature_store/run_feature_store.py

Import path strategy
--------------------
Both streaming/ and feature_store/ are siblings under the project root.
Each has its own sibling-import convention (bare 'from xxx import').

To avoid the two config files colliding on the "config" module slot:
  - streaming uses  config.py     → imported as "config"
  - feature_store uses fs_config.py → imported as "fs_config"  (no clash)

sys.path order:
  [0] feature_store/   ← feature_store modules find fs_config, user_store, etc.
  [1] streaming/       ← streaming modules find config, queue_manager, producer, etc.
"""

import asyncio
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STREAMING_DIR = os.path.join(_ROOT, "streaming")
_FS_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, _STREAMING_DIR)   # streaming siblings resolve first in their modules
sys.path.insert(0, _FS_DIR)          # feature_store siblings resolve first in their modules

from consumer import consume_events   # feature_store/consumer.py
from producer import produce          # streaming/producer.py
from queue_manager import register_consumer  # streaming/queue_manager.py

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


async def main() -> None:
    logger.info("Feature store initializing")
    queue = register_consumer("feature_store")
    await asyncio.gather(produce(), consume_events(queue))
    logger.info("Feature store complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[STOPPED] Feature store interrupted.")
