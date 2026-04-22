"""
FastAPI route handlers for the recommendation API.
All routes are registered on the router imported by main.py.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.event_logger import EventLogger
from api.schemas import (
    EventRequest,
    EventResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MovieDetailResponse,
    RecommendationResponse,
    RecommendedMovie,
    SearchResponse,
    TasteProfileResponse,
)
from serving.inference import RecommendationEngine

logger = logging.getLogger(__name__)
router = APIRouter()


# ------------------------------------------------------------------
# Dependency injection helpers (populated at startup in main.py)
# ------------------------------------------------------------------

_engine: Optional[RecommendationEngine] = None
_event_logger: Optional[EventLogger] = None


def set_engine(engine: RecommendationEngine) -> None:
    global _engine
    _engine = engine


def set_event_logger(el: EventLogger) -> None:
    global _event_logger
    _event_logger = el


def get_engine() -> RecommendationEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    return _engine


def get_logger() -> EventLogger:
    if _event_logger is None:
        raise HTTPException(status_code=503, detail="Event logger not ready")
    return _event_logger


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(engine: RecommendationEngine = Depends(get_engine)):
    return HealthResponse(
        status="ok",
        models_loaded=True,
        faiss_index_size=engine.faiss_index.num_items,
        num_users=engine.prep.num_users,
        num_movies=engine.prep.num_movies,
    )


# ------------------------------------------------------------------
# Recommendations
# ------------------------------------------------------------------

@router.get(
    "/recommendations/{user_id}",
    response_model=RecommendationResponse,
    tags=["recommendations"],
)
def get_recommendations(
    user_id: int,
    limit: int = Query(default=20, ge=1, le=100),
    engine: RecommendationEngine = Depends(get_engine),
    event_log: EventLogger = Depends(get_logger),
):
    """
    Return personalised movie recommendations for a user.
    Runs the full Two-Tower → DeepFM → MMR pipeline.
    """
    recs = engine.recommend(user_id, limit=limit)
    event_log.log(user_id, "recommendation_served", metadata={"count": len(recs)})

    rec_movies = [
        RecommendedMovie(
            movie_id=r["movie_id"],
            title=r["title"],
            year=r["year"],
            genres=r["genres"],
            score=r.get("score", 1.0),
            explanation=r.get("explanation", ""),
        )
        for r in recs
    ]
    return RecommendationResponse(
        user_id=user_id,
        recommendations=rec_movies,
        total=len(rec_movies),
    )


# ------------------------------------------------------------------
# Movie detail
# ------------------------------------------------------------------

@router.get(
    "/movies/{movie_id}",
    response_model=MovieDetailResponse,
    tags=["movies"],
)
def get_movie(
    movie_id: int,
    engine: RecommendationEngine = Depends(get_engine),
):
    """Return movie metadata and similar movies."""
    movie = engine.get_movie(movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")

    similar = engine.get_similar(movie_id, limit=10)

    return MovieDetailResponse(
        movie_id=movie["movie_id"],
        title=movie["title"],
        year=movie["year"],
        genres=movie["genres"],
        similar_movies=similar,
    )


# ------------------------------------------------------------------
# Popular / trending
# ------------------------------------------------------------------

@router.get("/movies/popular", tags=["movies"])
def get_popular(
    limit: int = Query(default=20, ge=1, le=100),
    engine: RecommendationEngine = Depends(get_engine),
):
    """Return a list of popular movies (used for cold-start / homepage hero)."""
    return engine._popular_fallback(limit)


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------

@router.get("/search", response_model=SearchResponse, tags=["search"])
def search(
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(default=20, ge=1, le=50),
    engine: RecommendationEngine = Depends(get_engine),
    event_log: EventLogger = Depends(get_logger),
):
    """Instant search over movie titles and genres."""
    results = engine.search(q, limit=limit)
    event_log.log(0, "search", metadata={"query": q})

    from api.schemas import Movie
    return SearchResponse(
        query=q,
        results=[
            Movie(
                movie_id=r["movie_id"],
                title=r["title"],
                year=r["year"],
                genres=r["genres"],
            )
            for r in results
        ],
        total=len(results),
    )


# ------------------------------------------------------------------
# Taste profile
# ------------------------------------------------------------------

@router.get(
    "/taste-profile/{user_id}",
    response_model=TasteProfileResponse,
    tags=["users"],
)
def get_taste_profile(
    user_id: int,
    engine: RecommendationEngine = Depends(get_engine),
):
    """Return a user's genre preferences and activity stats."""
    profile = engine.get_taste_profile(user_id)
    if "error" in profile:
        raise HTTPException(status_code=404, detail=profile["error"])
    return TasteProfileResponse(**profile)


# ------------------------------------------------------------------
# Feedback
# ------------------------------------------------------------------

@router.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
def post_feedback(
    body: FeedbackRequest,
    event_log: EventLogger = Depends(get_logger),
):
    """
    Record user feedback (thumbs up/down, mood, explicit rating).
    Events are forwarded to the streaming layer for model updates.
    """
    event_id = event_log.log(
        user_id=body.user_id,
        event_type="feedback",
        movie_id=body.movie_id,
        metadata={
            "action": body.action,
            "mood": body.mood,
            "rating": body.rating,
        },
    )
    return FeedbackResponse(success=True, message=f"Feedback recorded (event {event_id})")


# ------------------------------------------------------------------
# Event logging (generic)
# ------------------------------------------------------------------

@router.post("/events", response_model=EventResponse, tags=["events"])
def log_event(
    body: EventRequest,
    event_log: EventLogger = Depends(get_logger),
):
    """Log a generic user interaction event."""
    event_id = event_log.log(
        user_id=body.user_id,
        event_type=body.event_type,
        movie_id=body.movie_id,
        metadata=body.metadata,
    )
    return EventResponse(event_id=event_id, success=True)


# ------------------------------------------------------------------
# Similar movies (standalone endpoint)
# ------------------------------------------------------------------

@router.get("/movies/{movie_id}/similar", tags=["movies"])
def get_similar(
    movie_id: int,
    limit: int = Query(default=10, ge=1, le=50),
    engine: RecommendationEngine = Depends(get_engine),
):
    """Return movies similar to the given movie."""
    similar = engine.get_similar(movie_id, limit=limit)
    if not similar:
        raise HTTPException(status_code=404, detail="Movie not found or no similar movies")
    return {"movie_id": movie_id, "similar": similar, "total": len(similar)}
