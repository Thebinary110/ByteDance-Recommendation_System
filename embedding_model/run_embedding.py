"""
Embedding model orchestrator — streams events, trains embeddings, rebuilds FAISS index.

Flow:
  streaming producer → asyncio Queue → consumer
                                           ↓  every event
                                     train_on_event()
                                           ↓  every FAISS_REBUILD_INTERVAL events
                                     faiss_index.build()       ← rebuild fires FIRST
                                           ↓  every LOG_INTERVAL events
                                     log stats + recommend()   ← uses FAISS if built

Rebuild fires before the log check at the same count boundary (e.g. at 20k:
rebuild happens, then the LOG check fires and recommend() already uses FAISS).

Run from the project root:
    python embedding_model/run_embedding.py

sys.path order: [EMB_DIR, STREAMING_DIR, ...]
"""

import asyncio
import logging
import os
import sys
from asyncio import Queue

_ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EMB_DIR       = os.path.dirname(os.path.abspath(__file__))
_STREAMING_DIR = os.path.join(_ROOT, "streaming")

sys.path.insert(0, _STREAMING_DIR)
sys.path.insert(0, _EMB_DIR)

from emb_config import FAISS_REBUILD_INTERVAL, LOG_INTERVAL
from embedding_store import embedding_counts
from faiss_index import faiss_index
from inference import recommend
from producer import produce
from queue_manager import register_consumer
from trainer import train_on_event

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


async def _consume(queue: Queue) -> None:
    count      = 0
    total_loss = 0.0

    logger.info("Embedding consumer starting")

    while True:
        event = await queue.get()

        if event is None:
            queue.task_done()
            break

        loss        = train_on_event(event)
        total_loss += loss
        count      += 1

        # Rebuild fires BEFORE the log check so recommend() at the same
        # count boundary already queries the freshly built index.
        if count % FAISS_REBUILD_INTERVAL == 0:
            n = faiss_index.build()
            logger.info(f"[FAISS] index rebuilt — {n:,} items indexed")

        if count % LOG_INTERVAL == 0:
            avg_loss         = total_loss / LOG_INTERVAL
            n_users, n_items = embedding_counts()
            user_id          = event["user_id"]
            recs             = recommend(user_id)
            source           = "FAISS" if faiss_index.built else "linear"

            logger.info(
                f"[EMBED]  events={count:>10,}  avg_loss={avg_loss:.4f}  "
                f"users={n_users:,}  items={n_items:,}  retrieval={source}"
            )
            logger.info(f"[EMBED]  user={user_id!r}  →  {recs}")

            total_loss = 0.0

        queue.task_done()

    # Final snapshot
    n_users, n_items = embedding_counts()
    logger.info(
        f"[EMBED]  Done.  total_events={count:,}  "
        f"users={n_users:,}  items={n_items:,}"
    )


async def main() -> None:
    logger.info("Embedding model initializing (two-tower, online SGD + FAISS)")
    queue = register_consumer("embedding_model")
    await asyncio.gather(produce(), _consume(queue))
    logger.info("Embedding model complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[STOPPED] Embedding model interrupted.")
