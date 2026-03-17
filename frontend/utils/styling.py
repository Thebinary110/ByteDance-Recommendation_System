"""
Global CSS injection for the Streamlit dashboard.

Call apply_style() once at the top of app.py, before any other st.* calls.
"""

import streamlit as st

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── background ── */
.stApp {
    background-color: #0D1117;
    color: #C9D1D9;
}

/* ── hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── app header ── */
.app-header {
    border-bottom: 1px solid #1E2A3A;
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
}
.app-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    color: #8BA7C7;
    text-transform: uppercase;
    margin: 0;
}
.app-subtitle {
    font-size: 0.75rem;
    color: #3D5166;
    letter-spacing: 0.12em;
    margin-top: 0.25rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── section labels ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    color: #3D5166;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    margin-top: 0;
}

/* ── divider ── */
.section-divider {
    border: none;
    border-top: 1px solid #161B27;
    margin: 1.8rem 0;
}

/* ── recommendation card ── */
.rec-card {
    background: #0F1722;
    border: 1px solid #1A2535;
    border-left: 3px solid #1E3A5F;
    border-radius: 4px;
    padding: 0.55rem 0.85rem;
    margin-bottom: 0.45rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.rec-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #3D5166;
    min-width: 2rem;
}
.rec-item {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #8BA7C7;
}

/* ── tag chips (likes / dislikes) ── */
.tag-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.4rem;
}
.tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #5A7FA0;
    background: #0F1722;
    border: 1px solid #1A2535;
    border-radius: 3px;
    padding: 0.18rem 0.5rem;
    letter-spacing: 0.04em;
}

/* ── stat card ── */
.stat-card {
    background: #0F1722;
    border: 1px solid #1A2535;
    border-radius: 4px;
    padding: 0.9rem 1.1rem;
    text-align: center;
}
.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: #8BA7C7;
    display: block;
}
.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    color: #3D5166;
    text-transform: uppercase;
    margin-top: 0.3rem;
    display: block;
}

/* ── selectbox ── */
.stSelectbox label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: #3D5166;
    text-transform: uppercase;
}
div[data-baseweb="select"] > div {
    background: #0F1722;
    border: 1px solid #1E2A3A;
    border-radius: 3px;
    color: #8BA7C7;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}

/* ── empty state ── */
.empty-state {
    font-size: 0.8rem;
    color: #2D3D50;
    font-style: italic;
    padding: 0.8rem 0;
    letter-spacing: 0.04em;
}

/* ── column padding ── */
[data-testid="column"] {
    padding-left: 0.3rem;
    padding-right: 0.3rem;
}
"""


def apply_style() -> None:
    """Inject global CSS into the Streamlit page."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
