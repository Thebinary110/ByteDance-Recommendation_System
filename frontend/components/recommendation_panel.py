"""
Recommendation panel — renders enriched + explained movie recommendations.

Each card shows:
  - Poster thumbnail (left column, from TMDB)
  - Rank + movie title + IMDb link (right column)
  - Genre chips (shared genres highlighted in accent blue)
  - Reason line:   "because you liked: X (88%) · Y (76%)"
  - Semantic line: "conceptually similar to: The Matrix · Dark City"
"""

import streamlit as st
from typing import Dict, List

_FALLBACK_POSTER = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='80' height='120' "
    "style='background:%230B111A'>"
    "<text x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' "
    "font-size='11' fill='%231E3A5F'>no poster</text>"
    "</svg>"
)


def render_recommendations(recommendations: List[Dict]) -> None:
    st.markdown('<p class="section-label">Recommended Items</p>', unsafe_allow_html=True)

    if not recommendations:
        st.markdown(
            '<p class="empty-state">no recommendations yet — model warming up (FAISS builds at 20k events)</p>',
            unsafe_allow_html=True,
        )
        return

    shared_set_per_item = [set(item.get("shared_genres", [])) for item in recommendations]

    cards_html = ""
    for rank, (item, shared_set) in enumerate(zip(recommendations, shared_set_per_item), start=1):
        title           = item.get("title",   f"Unknown ({item.get('item_id', '?')})")
        genres          = item.get("genres",  [])
        reason          = item.get("reason",  [])
        reason_scores   = item.get("reason_scores", [])
        semantic_reason = item.get("semantic_reason", [])
        poster          = item.get("poster")   or _FALLBACK_POSTER
        imdb_url        = item.get("imdb_url")

        # --- Genre chips ---
        genre_chips = ""
        for g in genres:
            if g in shared_set:
                genre_chips += (
                    f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:0.60rem; '
                    f'color:#8BA7C7; background:#0F1F35; border:1px solid #1E3A5F; '
                    f'border-radius:3px; padding:0.12rem 0.4rem; margin-right:0.25rem; '
                    f'display:inline-block; margin-bottom:0.2rem;">{g}</span>'
                )
            else:
                genre_chips += (
                    f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:0.60rem; '
                    f'color:#3D5166; background:#0F1722; border:1px solid #1A2535; '
                    f'border-radius:3px; padding:0.12rem 0.4rem; margin-right:0.25rem; '
                    f'display:inline-block; margin-bottom:0.2rem;">{g}</span>'
                )

        # --- Reason line ---
        reason_html = ""
        if reason:
            reason_parts = []
            for title_r, score in zip(reason, reason_scores):
                reason_parts.append(
                    f"{title_r} <span style='color:#3D5166;'>({score}%)</span>"
                )
            reason_html = (
                f'<div style="margin-top:0.4rem;">'
                f'  <span style="font-family:\'JetBrains Mono\',monospace; font-size:0.60rem; '
                f'color:#5A7FA0; letter-spacing:0.03em;">'
                f'because you liked: {" &nbsp;·&nbsp; ".join(reason_parts)}'
                f'  </span>'
                f'</div>'
            )

        # --- Semantic reason line ---
        semantic_html = ""
        if semantic_reason:
            semantic_html = (
                f'<div style="margin-top:0.25rem;">'
                f'  <span style="font-family:\'JetBrains Mono\',monospace; font-size:0.60rem; '
                f'color:#C8963E; letter-spacing:0.03em;">'
                f'conceptually similar to: {" &nbsp;·&nbsp; ".join(semantic_reason)}'
                f'  </span>'
                f'</div>'
            )

        # --- IMDb link ---
        imdb_html = ""
        if imdb_url:
            imdb_html = (
                f'<div style="margin-top:0.45rem;">'
                f'  <a href="{imdb_url}" target="_blank" '
                f'     style="font-family:\'JetBrains Mono\',monospace; font-size:0.60rem; '
                f'color:#4A6741; text-decoration:none; border:1px solid #2A3D28; '
                f'border-radius:3px; padding:0.1rem 0.4rem;">'
                f'IMDb'
                f'  </a>'
                f'</div>'
            )

        # --- Card: poster left | details right ---
        cards_html += (
            f'<div class="rec-card" style="flex-direction:row; align-items:flex-start; '
            f'padding:0.65rem 0.85rem; margin-bottom:0.5rem; gap:0.85rem;">'

            # Poster column
            f'  <div style="flex-shrink:0;">'
            f'    <img src="{poster}" width="70" '
            f'         style="border-radius:4px; display:block; '
            f'background:#0B111A; min-height:105px;" '
            f'         onerror="this.style.display=\'none\'" />'
            f'  </div>'

            # Details column
            f'  <div style="flex:1; min-width:0;">'
            f'    <div style="display:flex; align-items:center; gap:0.6rem;">'
            f'      <span class="rec-rank">#{rank:02d}</span>'
            f'      <span style="font-family:\'Inter\',sans-serif; font-size:0.88rem; '
            f'font-weight:500; color:#C9D1D9; white-space:nowrap; overflow:hidden; '
            f'text-overflow:ellipsis;">{title}</span>'
            f'    </div>'
            f'    <div style="margin-top:0.35rem;">{genre_chips}</div>'
            f'    {reason_html}'
            f'    {semantic_html}'
            f'    {imdb_html}'
            f'  </div>'

            f'</div>'
        )

    st.markdown(cards_html, unsafe_allow_html=True)
