"""
Recommendation panel — renders the MLP-ranked final top-10 item list.
"""

import streamlit as st
from typing import List


def render_recommendations(recommendations: List[str]) -> None:
    st.markdown('<p class="section-label">Recommended Items</p>', unsafe_allow_html=True)

    if not recommendations:
        st.markdown(
            '<p class="empty-state">no recommendations yet — model warming up (FAISS builds at 20k events)</p>',
            unsafe_allow_html=True,
        )
        return

    cards_html = ""
    for rank, item_id in enumerate(recommendations, start=1):
        cards_html += (
            f'<div class="rec-card">'
            f'  <span class="rec-rank">#{rank:02d}</span>'
            f'  <span class="rec-item">{item_id}</span>'
            f'</div>'
        )

    st.markdown(cards_html, unsafe_allow_html=True)
