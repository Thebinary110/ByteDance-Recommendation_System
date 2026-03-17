from pydantic import BaseModel
from typing import List


class RecommendationResponse(BaseModel):
    user_id:         str
    recommendations: List[str]
    count:           int


class StatsResponse(BaseModel):
    total_users:      int
    total_items:      int
    total_emb_users:  int
    faiss_built:      bool
