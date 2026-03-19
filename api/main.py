"""
Real-Time Recommendation API — FastAPI entry point.

The ranking pipeline and feature-store consumer are started as daemon threads
inside the FastAPI lifespan so the API process shares in-memory state with
the backend (embedding store, feature store, ranking model).

Run:
    uvicorn api.main:app --reload --port 8000

Swagger UI:
    http://localhost:8000/docs
"""

import asyncio
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

# ── api/ must be on sys.path for intra-package imports (routes, services…) ───
_API_DIR = Path(__file__).parent
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

# core.config extends sys.path to reach feature_store/, embedding_model/, etc.
# This must happen before any service module is imported.
from core.config import API_TITLE, API_VERSION  # noqa: E402 (intentional order)
from utils.logger import logger
from routes.evaluation_routes import router as eval_router
from routes.recommendation_routes import router as rec_router
from routes.stats_routes import router as stats_router
from routes.user_routes import router as user_router


# ── pipeline startup ──────────────────────────────────────────────────────────

def _start_pipeline() -> None:
    """
    Start a single daemon thread that runs one asyncio event loop containing
    four concurrent coroutines:
      - produce()           — streams events from CSV into all queues
      - ranking_consume()   — trains embedding model + MLP ranker
      - deepfm_consume()    — trains DeepFM ranker on 'like' events
      - fs_consume()        — populates _store (likes / dislikes / history)

    All queues are created inside that event loop so there is no cross-loop
    asyncio.Queue conflict (which occurs when a Queue is created in uvicorn's
    loop and then awaited from a different thread's loop).
    """
    from queue_manager import register_consumer   # streaming
    from user_store import update_user            # feature_store
    from producer import produce                  # streaming
    from run_ranker import _consume as ranking_consume  # ranking_model
    from deepfm_trainer import train_step         # deepfm
    from embedding_store import user_embeddings, item_embeddings  # embedding_model

    async def _combined() -> None:
        ranking_queue = register_consumer("ranking_pipeline")
        fs_queue      = register_consumer("feature_store_api")
        deepfm_queue  = register_consumer("deepfm_pipeline")

        async def _fs_consume() -> None:
            while True:
                event = await fs_queue.get()
                if event is None:
                    fs_queue.task_done()
                    break
                update_user(
                    event["user_id"],
                    event["item_id"],
                    event["event_type"],
                    event["timestamp"],
                )
                fs_queue.task_done()

        async def _deepfm_consume() -> None:
            while True:
                event = await deepfm_queue.get()
                if event is None:
                    deepfm_queue.task_done()
                    break
                # Attach detached embeddings so train_step can call build_features
                uid = str(event["user_id"])
                iid = str(event["item_id"])
                u_tensor = user_embeddings.get(uid)
                i_tensor = item_embeddings.get(iid)
                if u_tensor is not None and i_tensor is not None:
                    enriched = dict(event)
                    enriched["user_emb"] = u_tensor.detach()
                    enriched["item_emb"] = i_tensor.detach()
                    train_step(enriched)
                deepfm_queue.task_done()

        await asyncio.gather(
            produce(),
            ranking_consume(ranking_queue),
            _fs_consume(),
            _deepfm_consume(),
        )

    def _run() -> None:
        asyncio.run(_combined())

    threading.Thread(target=_run, name="ranking-pipeline", daemon=True).start()
    logger.info("Pipeline thread started (ranking + DeepFM + feature-store in one event loop)")


# ── lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _start_pipeline()
    logger.info("API startup complete — pipeline running in background")
    yield
    logger.info("API shutdown")


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Two-Tower Embedding + FAISS Retrieval + MLP Reranking",
    lifespan=lifespan,
)

app.include_router(user_router)
app.include_router(rec_router)
app.include_router(stats_router)
app.include_router(eval_router)


@app.get("/", tags=["health"])
def root():
    return {"message": "Recommendation API running", "version": API_VERSION}
