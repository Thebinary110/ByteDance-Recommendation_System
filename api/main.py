"""
FastAPI application entry-point.

Local:
  uvicorn api.main:app --reload --port 8000

Render (set env vars in dashboard — HF_REPO_ID, HF_TOKEN, REDIS_URL):
  uvicorn api.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any os.getenv() calls — works locally and is a no-op on
# Render (where env vars are injected directly by the platform).
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
# Step 1 — Download
# ------------------------------------------------------------------

def _download_artifacts_from_hf(repo_id: str, artifact_dir: Path) -> None:
    """
    Pull every missing serving artifact from a HuggingFace Hub model repo.

    Args:
        repo_id      : e.g. "IntimateUser6969/movielens-recommender"
        artifact_dir : local directory to save files into

    Auth: reads HF_TOKEN from env (set in Render dashboard or .env).
    Only downloads files that are not already on disk, so restarts after the
    first boot are instant.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("Run: pip install huggingface_hub")

    token = os.getenv("HF_TOKEN") or None  # None → uses cached login token
    artifact_dir.mkdir(parents=True, exist_ok=True)

    missing = [f for f in _REQUIRED_ARTIFACTS if not (artifact_dir / f).exists()]
    if not missing:
        logger.info("All artifacts already on disk — skipping HuggingFace download.")
        return

    logger.info(f"Downloading {len(missing)} artifact(s) from '{repo_id}' ...")
    for filename in missing:
        logger.info(f"  Fetching {filename}")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir=str(artifact_dir),
            local_dir_use_symlinks=False,
            token=token,
        )
        logger.info(f"    saved to {artifact_dir / filename}")

    logger.info("HuggingFace download complete.")


# ------------------------------------------------------------------
# Step 2 — Check
# ------------------------------------------------------------------

def _check_artifacts(artifact_dir: Path) -> None:
    """Raise a descriptive RuntimeError if any required file is missing."""
    missing = [f for f in _REQUIRED_ARTIFACTS if not (artifact_dir / f).exists()]
    if missing:
        raise RuntimeError(
            f"\n\n{'='*60}\n"
            f"  Artifacts missing in: {artifact_dir}\n"
            f"  Missing: {', '.join(missing)}\n\n"
            f"  Option A — train locally:\n"
            f"    ./run_training.sh\n"
            f"  Option B — set HF_REPO_ID (+ HF_TOKEN for private repos)\n"
            f"    so the API downloads them on first boot.\n"
            f"{'='*60}\n"
        )


# ------------------------------------------------------------------
# Step 3 — Load
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
# Lifespan — wires the three steps together
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting up — artifact dir: {ARTIFACT_DIR}")

    # Step 1: Download from HuggingFace if running on Render / cold container
    hf_repo = os.getenv("HF_REPO_ID")
    if hf_repo:
        _download_artifacts_from_hf(hf_repo, ARTIFACT_DIR)

    # Step 2: Verify all required files are present
    _check_artifacts(ARTIFACT_DIR)

    # Step 3: Load models into memory
    engine, event_logger = load_models(ARTIFACT_DIR)
    set_engine(engine)
    set_event_logger(event_logger)

    logger.info("Startup complete — ready to serve.")
    yield

    logger.info("Shutting down ...")
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

# CORS: accept local dev ports + any *.onrender.com subdomain.
# CORS_ORIGINS env var lets Render / CI override this without a code change.
_extra_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

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
        *_extra_origins,
    ],
    allow_origin_regex=r"https://.*\.onrender\.com",  # matches any Render deploy URL
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
