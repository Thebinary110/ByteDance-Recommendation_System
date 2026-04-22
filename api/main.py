"""
FastAPI application entry-point.

Start with:
  uvicorn api.main:app --reload --port 8000

The app loads all ML models and indexes at startup so every request
is served from in-memory state with no cold-start penalty.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.event_logger import EventLogger
from api.routes import router, set_engine, set_event_logger
from serving.feature_store import FeatureStore
from serving.inference import RecommendationEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Resolve artifacts relative to the recommender/ root (parent of api/).
# Using __file__ makes this work regardless of which directory uvicorn is
# launched from — the relative "artifacts" default used to break when the
# script was started outside recommender/.
_RECOMMENDER_ROOT = Path(__file__).resolve().parent.parent  # recommender/
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", str(_RECOMMENDER_ROOT / "artifacts")))


def _check_artifacts(artifact_dir: Path) -> None:
    """Raise a clear RuntimeError if required artifact files are missing."""
    required = ["preprocessor.pkl", "two_tower.pt", "deepfm_best.pt",
                "faiss.index", "item_embeddings.npy"]
    missing = [f for f in required if not (artifact_dir / f).exists()]
    if missing:
        raise RuntimeError(
            f"\n\n{'='*60}\n"
            f"  Artifacts not found in: {artifact_dir}\n"
            f"  Missing: {', '.join(missing)}\n\n"
            f"  Run training first:\n"
            f"    cd {_RECOMMENDER_ROOT}\n"
            f"    python -m training.train_two_tower --data_dir ../ --output_dir artifacts/ --sample_frac 0.1\n"
            f"    python -m training.train_deepfm    --output_dir artifacts/\n"
            f"  (or run ./run_training.sh)\n"
            f"{'='*60}\n"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    logger.info(f"Loading artifacts from {ARTIFACT_DIR} …")
    _check_artifacts(ARTIFACT_DIR)

    feature_store = FeatureStore(
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", 6379)),
        sqlite_path=ARTIFACT_DIR / "feature_store.db",
    )

    engine = RecommendationEngine.load(
        ARTIFACT_DIR,
        device_str=os.getenv("DEVICE", "cpu"),
        feature_store=feature_store,
    )
    set_engine(engine)

    event_logger = EventLogger(
        kafka_bootstrap=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
        sqlite_path=ARTIFACT_DIR / "events.db",
    )
    set_event_logger(event_logger)

    logger.info("Startup complete — serving requests.")
    yield

    logger.info("Shutting down …")
    event_logger.close()


app = FastAPI(
    title="CineMatch Recommendation API",
    description=(
        "Production recommendation system powered by Two-Tower candidate retrieval, "
        "DeepFM ranking, and MMR diversity re-ranking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the React dev server (port 5173) and any same-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://localhost:5177",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5177",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/", tags=["system"])
def root():
    return {
        "service": "CineMatch Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
    }
