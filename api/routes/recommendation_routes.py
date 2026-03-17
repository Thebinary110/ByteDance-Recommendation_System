from fastapi import APIRouter, HTTPException, Query

from schemas.recommendation_schema import RecommendationResponse
from services.recommendation_service import get_recommendations
from services.user_service import get_all_users

router = APIRouter(prefix="/recommend", tags=["recommendations"])


@router.get("/{user_id}", response_model=RecommendationResponse)
def recommend(
    user_id: str,
    top_k: int = Query(default=10, ge=1, le=50, description="Number of items to return"),
):
    """
    Return top-k reranked recommendations for user_id.
    Uses FAISS retrieval (top-50 candidates) → MLP reranking → top-k.
    """
    if user_id not in get_all_users():
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")
    recs = get_recommendations(user_id, top_k=top_k)
    return {"user_id": user_id, "recommendations": recs, "count": len(recs)}
