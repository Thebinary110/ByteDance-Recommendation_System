"""
Recommendation list component — renders item cards with action buttons.
"""

import asyncio
from typing import Any

import streamlit as st

from services.interaction_service import record_interaction


def render_recommendations(
    recommendations: list[dict[str, Any]],
    user_id: int,
) -> None:
    st.markdown(
        '<p class="section-label">RECOMMENDATIONS</p>',
        unsafe_allow_html=True,
    )

    if not recommendations:
        st.markdown(
            '<p class="empty-state">No recommendations loaded. Enter a User ID and click Load Recommendations.</p>',
            unsafe_allow_html=True,
        )
        return

    for item in recommendations:
        _render_item_card(item, user_id)


def _render_item_card(item: dict[str, Any], user_id: int) -> None:
    item_id: int = item["item_id"]
    title: str = item["title"]

    with st.container():
        st.markdown(f'<div class="item-card">', unsafe_allow_html=True)

        col_meta, col_like, col_dislike = st.columns([6, 1, 1], gap="small")

        with col_meta:
            st.markdown(
                f'<span class="item-id">#{item_id}</span>'
                f'<span class="item-title">{title}</span>',
                unsafe_allow_html=True,
            )

        with col_like:
            if st.button(
                "Like",
                key=f"like_{item_id}",
                use_container_width=True,
            ):
                asyncio.run(
                    record_interaction(user_id, item_id, title, "like")
                )
                st.toast(f"Liked: {title}", icon=None)

        with col_dislike:
            if st.button(
                "Dislike",
                key=f"dislike_{item_id}",
                use_container_width=True,
            ):
                asyncio.run(
                    record_interaction(user_id, item_id, title, "dislike")
                )
                st.toast(f"Disliked: {title}", icon=None)

        st.markdown("</div>", unsafe_allow_html=True)
