from pydantic import BaseModel
from typing import List


class RecommendedItem(BaseModel):
    item_id:       str
    title:         str
    genres:        List[str]
    reason:        List[str]         # titles of similar liked items
    reason_scores: List[float]       # cosine similarity as % (bonus)
    shared_genres: List[str]         # genres overlapping with user's likes (bonus)


class RecommendationResponse(BaseModel):
    user_id:         str
    recommendations: List[RecommendedItem]
    count:           int


class StatsResponse(BaseModel):
    total_users:     int
    total_items:     int
    total_emb_users: int
    faiss_built:     bool
