from fastapi import APIRouter

from schemas.recommendation_schema import StatsResponse
from services.stats_service import get_stats

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("", response_model=StatsResponse)
def stats():
    """Return system-level metrics: user count, item count, FAISS status."""
    return get_stats()
