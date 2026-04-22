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

from dotenv import load_dotenv

# Load .env from the recommender/ root before any os.getenv() calls
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

# Resolve artifacts relative to the recommender/ root (parent of api/).
# Using __file__ makes this work regardless of which directory uvicorn is
# launched from — the relative "artifacts" default used to break when the
# script was started outside recommender/.
_RECOMMENDER_ROOT = Path(__file__).resolve().parent.parent  # recommender/
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", str(_RECOMMENDER_ROOT / "artifacts")))


# All artifact filenames that must be present for the engine to start.
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


def _check_artifacts(artifact_dir: Path) -> None:
    """Raise a clear RuntimeError if required artifact files are missing."""
    missing = [f for f in _REQUIRED_ARTIFACTS if not (artifact_dir / f).exists()]
    if missing:
        raise RuntimeError(
            f"\n\n{'='*60}\n"
            f"  Artifacts not found in: {artifact_dir}\n"
            f"  Missing: {', '.join(missing)}\n\n"
            f"  Option A — run training locally:\n"
            f"    cd {_RECOMMENDER_ROOT}\n"
            f"    python -m training.train_two_tower --data_dir ../ --output_dir artifacts/ --sample_frac 0.1\n"
            f"    python -m training.train_deepfm    --output_dir artifacts/\n"
            f"  Option B — set HF_REPO_ID env var to download from HuggingFace Hub:\n"
            f"    HF_REPO_ID=your-username/movielens-recommender uvicorn api.main:app\n"
            f"{'='*60}\n"
        )


def _download_artifacts_from_hf(artifact_dir: Path, repo_id: str) -> None:
    """
    Download missing model artifacts from a HuggingFace Hub model repository.

    Only files that don't already exist locally are fetched, so re-starts are
    instant once the cache is warm.  Set HF_REPO_ID to enable this path.

    Upload your artifacts first:
        pip install huggingface_hub
        huggingface-cli login
        python - <<'EOF'
        from huggingface_hub import HfApi
        HfApi().upload_folder(
            folder_path="artifacts/",
            repo_id="your-username/movielens-recommender",
            repo_type="model",
        )
        EOF
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is not installed. "
            "Run: pip install huggingface_hub"
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    missing = [f for f in _REQUIRED_ARTIFACTS if not (artifact_dir / f).exists()]

    if not missing:
        logger.info("All artifacts already present — skipping HuggingFace download.")
        return

    logger.info(
        f"Downloading {len(missing)} artifact(s) from HuggingFace Hub "
        f"repo '{repo_id}' …"
    )
    for filename in missing:
        logger.info(f"  ↓ {filename}")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir=str(artifact_dir),
            local_dir_use_symlinks=False,
        )
        logger.info(f"    saved → {local_path}")

    logger.info("HuggingFace artifact download complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    logger.info(f"Loading artifacts from {ARTIFACT_DIR} …")

    # If HF_REPO_ID is set, pull any missing artifacts from HuggingFace Hub
    # before the existence check.  This is the deployment path; local runs
    # just skip this block entirely.
    hf_repo = os.getenv("HF_REPO_ID")
    if hf_repo:
        _download_artifacts_from_hf(ARTIFACT_DIR, hf_repo)

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
