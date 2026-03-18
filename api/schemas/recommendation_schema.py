from pydantic import BaseModel
from typing import List, Optional


class RecommendedItem(BaseModel):
    item_id:         str
    title:           str
    genres:          List[str]
    reason:          List[str]           # titles of similar liked items (embedding cosine)
    reason_scores:   List[float]         # cosine similarity as %
    shared_genres:   List[str]           # genres overlapping with user's likes
    semantic_reason: List[str]           # titles of conceptually similar liked items (genome tags)
    poster:          Optional[str]       # TMDB poster image URL (None if unavailable)
    imdb_url:        Optional[str]       # full IMDb title URL (None if unavailable)


class RecommendationResponse(BaseModel):
    user_id:         str
    recommendations: List[RecommendedItem]
    count:           int


class StatsResponse(BaseModel):
    total_users:     int
    total_items:     int
    total_emb_users: int
    faiss_built:     bool
