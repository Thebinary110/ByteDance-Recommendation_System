"""
Full ranking pipeline orchestrator.

Two-stage recommendation flow per event:
  Stage A — Embedding model:  train two-tower vectors + FAISS retrieval
  Stage B — Ranking model:    MLP reranks FAISS candidates → final Top-K

    streaming producer
          ↓
      asyncio Queue
          ↓  every event
    train_on_event()       ← embedding model (two-tower + negative sampling)
    train_ranker()         ← MLP ranking model (Adam)
          ↓  every FAISS_REBUILD_INTERVAL events
    faiss_index.build()    ← refresh ANN index
          ↓  every LOG_INTERVAL events
    retrieve(user, 50)     ← FAISS top-50 candidates
    rank_items(user, 50)   ← MLP top-10 final recs
    log losses + recs

Run from the project root:
    python ranking_model/run_ranker.py

Import path strategy
--------------------
Three directories on sys.path:
  ranking_model/  → rank_config, rank_model, rank_trainer, rank_inference
  embedding_model/ → emb_config, embedding_store, model, faiss_index,
                     inference, trainer, negative_sampler
  streaming/       → config, queue_manager, producer

No "config.py" in ranking_model/ or embedding_model/ so streaming/config.py
takes the "config" slot without conflict.
"""

import asyncio
import logging
import os
import sys
from asyncio import Queue

_ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RANKING_DIR   = os.path.dirname(os.path.abspath(__file__))
_EMB_DIR       = os.path.join(_ROOT, "embedding_model")
_STREAMING_DIR = os.path.join(_ROOT, "streaming")

sys.path.insert(0, _STREAMING_DIR)
sys.path.insert(0, _EMB_DIR)
sys.path.insert(0, _RANKING_DIR)

from embedding_store import embedding_counts
from faiss_index import faiss_index
from inference import recommend as retrieve    # FAISS-backed retrieval
from producer import produce
from queue_manager import register_consumer
from rank_config import (
    FAISS_REBUILD_INTERVAL,
    LOG_INTERVAL,
    TOP_K_CANDIDATES,
    TOP_K_FINAL,
)
from rank_inference import rank_items
from rank_trainer import train_ranker
from trainer import train_on_event            # embedding model trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


async def _consume(queue: Queue) -> None:
    count      = 0
    emb_loss   = 0.0
    rank_loss  = 0.0

    logger.info("Ranking pipeline consumer starting")

    while True:
        event = await queue.get()

        if event is None:
            queue.task_done()
            break

        emb_loss  += train_on_event(event)   # update two-tower embeddings
        rank_loss += train_ranker(event)      # update MLP ranker
        count     += 1

        # FAISS rebuild fires before log so the recommendation at the same
        # count boundary already queries the refreshed index.
        if count % FAISS_REBUILD_INTERVAL == 0:
            n = faiss_index.build()
            logger.info(f"[FAISS] index rebuilt — {n:,} items")

        if count % LOG_INTERVAL == 0:
            avg_emb  = emb_loss  / LOG_INTERVAL
            avg_rank = rank_loss / LOG_INTERVAL
            n_users, n_items = embedding_counts()
            user_id  = event["user_id"]

            candidates = retrieve(user_id, top_k=TOP_K_CANDIDATES)
            ranked     = rank_items(user_id, candidates, top_k=TOP_K_FINAL)

            logger.info(
                f"[RANKER]  events={count:>10,}  "
                f"emb_loss={avg_emb:.4f}  rank_loss={avg_rank:.4f}  "
                f"users={n_users:,}  items={n_items:,}"
            )
            logger.info(f"[FINAL REC]  user={user_id!r}  →  {ranked}")

            emb_loss  = 0.0
            rank_loss = 0.0

        queue.task_done()

    n_users, n_items = embedding_counts()
    logger.info(
        f"[RANKER]  Done.  total_events={count:,}  "
        f"users={n_users:,}  items={n_items:,}"
    )


async def main() -> None:
    logger.info(
        "Ranking pipeline initializing "
        "(embedding model + FAISS + MLP ranker)"
    )
    queue = register_consumer("ranking_pipeline")
    await asyncio.gather(produce(), _consume(queue))
    logger.info("Ranking pipeline complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[STOPPED] Ranking pipeline interrupted.")
