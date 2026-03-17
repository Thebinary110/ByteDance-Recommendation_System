"""
Interaction service — records user actions and fetches recommendations.
All functions are async-ready for seamless backend integration in Stage 2.
"""

import asyncio
from typing import Any

import streamlit as st

from utils.helpers import generate_dummy_recommendations, get_timestamp


async def fetch_recommendations(user_id: int) -> list[dict[str, Any]]:
    """
    Async stub — returns dummy recommendations.
    Stage 2: replace body with:
        response = await httpx.AsyncClient().get(f"/api/recommendations/{user_id}")
        return response.json()
    """
    await asyncio.sleep(0)  # yield point preserved for real async integration
    return generate_dummy_recommendations(user_id)


async def record_interaction(
    user_id: int,
    item_id: int,
    title: str,
    action: str,
) -> dict[str, Any]:
    """
    Async stub — stores interaction in session state.
    Stage 2: replace body with a POST to the interaction API.
    """
    await asyncio.sleep(0)
    entry = {
        "user_id": user_id,
        "item_id": item_id,
        "title": title,
        "action": action,
        "timestamp": get_timestamp(),
    }
    st.session_state.interaction_log.append(entry)
    return entry
