"""
User panel — displays a user's like/dislike profile and last interaction time.

Receives a plain dict from the API (likes/dislikes are already List[str]).
"""

import streamlit as st
from typing import Dict


def render_user_panel(user: Dict) -> None:
    st.markdown('<p class="section-label">User Profile</p>', unsafe_allow_html=True)

    likes    = list(user.get("likes",    []))[:20]
    dislikes = list(user.get("dislikes", []))[:20]
    last_ts  = user.get("last_interaction")
    history  = user.get("history_size", 0)

    meta_parts = []
    if last_ts:
        meta_parts.append(f"last interaction: {last_ts}")
    if history:
        meta_parts.append(f"history: {history} events")

    if meta_parts:
        st.markdown(
            f'<p style="font-family:\'JetBrains Mono\',monospace; font-size:0.68rem; '
            f'color:#3D5166; margin-bottom:0.8rem;">{" &nbsp;|&nbsp; ".join(meta_parts)}</p>',
            unsafe_allow_html=True,
        )

    col_likes, col_dislikes = st.columns(2)

    with col_likes:
        st.markdown('<p class="section-label">Likes</p>', unsafe_allow_html=True)
        if likes:
            chips = "".join(f'<span class="tag">{iid}</span>' for iid in likes)
            st.markdown(f'<div class="tag-grid">{chips}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="empty-state">no likes recorded</p>', unsafe_allow_html=True)

    with col_dislikes:
        st.markdown('<p class="section-label">Dislikes</p>', unsafe_allow_html=True)
        if dislikes:
            chips = "".join(f'<span class="tag">{iid}</span>' for iid in dislikes)
            st.markdown(f'<div class="tag-grid">{chips}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="empty-state">no dislikes recorded</p>', unsafe_allow_html=True)
