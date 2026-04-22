"""Pydantic schemas for request/response validation."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Movie schemas
# ------------------------------------------------------------------

class Movie(BaseModel):
    movie_id: int
    title: str
    year: int
    genres: list[str]


class RecommendedMovie(Movie):
    score: float
    explanation: str


# ------------------------------------------------------------------
# Recommendation response
# ------------------------------------------------------------------

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[RecommendedMovie]
    total: int


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------

class SearchResponse(BaseModel):
    query: str
    results: list[Movie]
    total: int


# ------------------------------------------------------------------
# Movie detail
# ------------------------------------------------------------------

class MovieDetailResponse(Movie):
    similar_movies: list[Movie] = []
    avg_rating: Optional[float] = None
    rating_count: Optional[int] = None


# ------------------------------------------------------------------
# Taste profile
# ------------------------------------------------------------------

class TasteProfileResponse(BaseModel):
    user_id: int
    genre_preferences: dict[str, float]
    avg_rating: float
    rating_count: int
    top_genres: list[str]


# ------------------------------------------------------------------
# Feedback
# ------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    user_id: int
    movie_id: int
    action: str = Field(..., description="One of: thumbs_up, thumbs_down, skip, watch")
    mood: Optional[str] = Field(None, description="Optional mood tag: happy, sad, excited, etc.")
    rating: Optional[float] = Field(None, ge=1.0, le=5.0)


class FeedbackResponse(BaseModel):
    success: bool
    message: str


# ------------------------------------------------------------------
# Event logging
# ------------------------------------------------------------------

class EventRequest(BaseModel):
    user_id: int
    event_type: str  # view, click, rating, search, feedback
    movie_id: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None


class EventResponse(BaseModel):
    event_id: str
    success: bool


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    faiss_index_size: int
    num_users: int
    num_movies: int
