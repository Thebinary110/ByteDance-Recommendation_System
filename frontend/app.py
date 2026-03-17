"""
Real-Time Recommendation System — Streamlit Dashboard (API-connected).

Architecture:
    Streamlit UI  →  HTTP  →  FastAPI (http://localhost:8000)  →  ML backend

The frontend is fully stateless — no backend imports, no pipeline threads.
All state lives in the API process.

Run:
    # Terminal 1 — API + pipeline
    uvicorn api.main:app --port 8000

    # Terminal 2 — Frontend
    streamlit run frontend/app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

# ── frontend path ─────────────────────────────────────────────────────────────
_FRONTEND = Path(__file__).parent
if str(_FRONTEND) not in sys.path:
    sys.path.insert(0, str(_FRONTEND))

from components.recommendation_panel import render_recommendations
from components.stats_panel import render_stats
from components.user_panel import render_user_panel
from services.data_service import (
    api_latency_ms,
    get_all_users,
    get_recommendations,
    get_stats,
    get_user_data,
)
from utils.styling import apply_style

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recommendation System",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_style()

REFRESH_INTERVAL = 5   # seconds between auto-refreshes

# ── header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <p class="app-title">Recommendation System</p>
        <p class="app-subtitle">
            Streamlit &nbsp;&mdash;&nbsp; FastAPI &nbsp;&mdash;&nbsp;
            Two-Tower &nbsp;&mdash;&nbsp; FAISS &nbsp;&mdash;&nbsp; MLP Reranking
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── wait for API to be reachable ──────────────────────────────────────────────
users = get_all_users()

if not users:
    st.markdown(
        """
        <div style="padding: 2rem 0; text-align: center;">
            <p style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;
                      color:#3D5166; letter-spacing:0.12em;">
                waiting for API &mdash; ensure the backend is running
            </p>
            <p style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
                      color:#2D3D50; letter-spacing:0.08em; margin-top:0.5rem;">
                uvicorn api.main:app --port 8000
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(1)
    st.rerun()
    st.stop()

# ── toolbar: user selector | manual refresh | latency ─────────────────────────
col_select, col_btn, col_lat = st.columns([5, 1, 1])

with col_select:
    selected_user = st.selectbox("Select User", users, label_visibility="visible")

with col_btn:
    st.markdown("<div style='margin-top:1.6rem'>", unsafe_allow_html=True)
    manual_refresh = st.button("Refresh")
    st.markdown("</div>", unsafe_allow_html=True)

with col_lat:
    latency = api_latency_ms()
    st.markdown(
        f'<div style="margin-top:1.9rem; text-align:right;">'
        f'  <span style="font-family:\'JetBrains Mono\',monospace; font-size:0.65rem; '
        f'color:#3D5166; letter-spacing:0.12em;">{latency} ms</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── fetch live data from API ──────────────────────────────────────────────────
user_data       = get_user_data(selected_user)
recommendations = get_recommendations(selected_user)
stats           = get_stats()

# ── layout: recommendations (left) | user profile (right) ────────────────────
col_recs, col_user = st.columns([2, 3], gap="large")

with col_recs:
    render_recommendations(recommendations)

with col_user:
    render_user_panel(user_data)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── stats row ─────────────────────────────────────────────────────────────────
render_stats(stats)

# ── footer: pipeline status + auto-refresh ───────────────────────────────────
faiss_ready = stats.get("faiss_built", False)
status_color = "#2A4A2A" if faiss_ready else "#4A3A1A"
status_msg   = "FAISS ready — full pipeline active" if faiss_ready else "pipeline warming up — FAISS builds at 20k events"

st.markdown(
    f'<p style="font-family:\'JetBrains Mono\',monospace; font-size:0.6rem; '
    f'color:{status_color}; letter-spacing:0.12em; margin-top:1.5rem; text-align:right;">'
    f'{status_msg}</p>',
    unsafe_allow_html=True,
)

# ── auto-refresh ──────────────────────────────────────────────────────────────
time.sleep(REFRESH_INTERVAL)
st.rerun()
