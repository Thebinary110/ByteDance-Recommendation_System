"""
Stats panel — system-level metrics fetched from the API.

Key names match the StatsResponse schema:
  total_users, total_items, total_emb_users, faiss_built
"""

import streamlit as st
from typing import Dict


def render_stats(stats: Dict) -> None:
    st.markdown('<p class="section-label">System Stats</p>', unsafe_allow_html=True)

    faiss_status = "built" if stats.get("faiss_built") else "warming up"
    faiss_color  = "#2A5C2A" if stats.get("faiss_built") else "#5C3A1E"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        _stat_card(stats.get("total_users", 0), "users")
    with col2:
        _stat_card(stats.get("total_items", 0), "item embeddings")
    with col3:
        _stat_card(stats.get("total_emb_users", 0), "user embeddings")
    with col4:
        st.markdown(
            f'<div class="stat-card">'
            f'  <span class="stat-value" style="font-size:1rem; color:{faiss_color};">'
            f'    {faiss_status}'
            f'  </span>'
            f'  <span class="stat-label">FAISS index</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _stat_card(value: int, label: str) -> None:
    st.markdown(
        f'<div class="stat-card">'
        f'  <span class="stat-value">{value:,}</span>'
        f'  <span class="stat-label">{label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
