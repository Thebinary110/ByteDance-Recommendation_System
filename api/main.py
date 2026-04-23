"""
FastAPI application entry-point.

Local:
  uvicorn api.main:app --reload --port 8000

HuggingFace Spaces:
  uvicorn api.main:app --host 0.0.0.0 --port 7860
  (set REDIS_URL in Space Settings → Secrets)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env locally — no-op on HuggingFace Spaces (env vars injected by platform)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

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

_RECOMMENDER_ROOT = Path(__file__).resolve().parent.parent  # recommender/
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", str(_RECOMMENDER_ROOT / "artifacts")))

_REQUIRED_ARTIFACTS = [
    "preprocessor.pkl",
    "movie_meta.csv",
    "two_tower.pt",
    "deepfm_best.pt",
    "faiss.index",
    "item_embeddings.npy",
    "item_features.npy",
    "user_features.npy",
]


# ------------------------------------------------------------------
# Step 1 — Check
# ------------------------------------------------------------------

def _check_artifacts(artifact_dir: Path) -> None:
    """Raise a descriptive RuntimeError if any required file is missing."""
    missing = [f for f in _REQUIRED_ARTIFACTS if not (artifact_dir / f).exists()]
    if missing:
        raise RuntimeError(
            f"\n\n{'='*60}\n"
            f"  Artifacts missing in: {artifact_dir}\n"
            f"  Missing: {', '.join(missing)}\n\n"
            f"  Run training to generate them:\n"
            f"    ./run_training.sh\n"
            f"{'='*60}\n"
        )


# ------------------------------------------------------------------
# Step 2 — Load
# ------------------------------------------------------------------

def load_models(artifact_dir: Path) -> tuple[RecommendationEngine, EventLogger]:
    """
    Instantiate the feature store, recommendation engine, and event logger.
    Kept as a plain function so it can be called from tests or CLI without
    going through the full FastAPI lifespan.
    """
    feature_store = FeatureStore(
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", 6379)),
        sqlite_path=artifact_dir / "feature_store.db",
    )

    engine = RecommendationEngine.load(
        artifact_dir,
        device_str=os.getenv("DEVICE", "cpu"),
        feature_store=feature_store,
    )

    event_logger = EventLogger(
        kafka_bootstrap=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
        sqlite_path=artifact_dir / "events.db",
    )

    return engine, event_logger


# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Artifact dir: {ARTIFACT_DIR}")
    _check_artifacts(ARTIFACT_DIR)
    engine, event_logger = load_models(ARTIFACT_DIR)
    set_engine(engine)
    set_event_logger(event_logger)
    logger.info("Startup complete.")
    yield
    event_logger.close()


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="CineMatch Recommendation API",
    description=(
        "Production recommendation system: Two-Tower retrieval, "
        "DeepFM ranking, MMR diversity re-ranking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: local dev ports + HuggingFace Space + any extra origins from env
_extra_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://intimateuser6969-cinewatch-recommender.hf.space",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://localhost:5177",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5177",
        *_extra_origins,
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
