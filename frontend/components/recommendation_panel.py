"""
Recommendation panel — renders enriched + explained movie recommendations.

Each card shows:
  - Rank + movie title
  - Genre chips (shared genres highlighted in accent blue)
  - Reason line: "Because you liked: X (88%), Y (76%)"
"""

import streamlit as st
from typing import Dict, List


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
        title         = item.get("title",   f"Unknown ({item.get('item_id', '?')})")
        genres        = item.get("genres",  [])
        reason        = item.get("reason",  [])
        reason_scores = item.get("reason_scores", [])

        # Genre chips — shared genres highlighted in accent blue
        genre_chips = ""
        for g in genres:
            if g in shared_set:
                genre_chips += (
                    f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:0.62rem; '
                    f'color:#8BA7C7; background:#0F1F35; border:1px solid #1E3A5F; '
                    f'border-radius:3px; padding:0.15rem 0.45rem; margin-right:0.3rem; '
                    f'display:inline-block;">{g}</span>'
                )
            else:
                genre_chips += (
                    f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:0.62rem; '
                    f'color:#3D5166; background:#0F1722; border:1px solid #1A2535; '
                    f'border-radius:3px; padding:0.15rem 0.45rem; margin-right:0.3rem; '
                    f'display:inline-block;">{g}</span>'
                )

        # Reason line
        reason_html = ""
        if reason:
            reason_parts = []
            for title_r, score in zip(reason, reason_scores):
                reason_parts.append(f"{title_r} <span style='color:#3D5166;'>({score}%)</span>")
            reason_html = (
                f'<div style="margin-top:0.45rem; padding-left:2.5rem;">'
                f'  <span style="font-family:\'JetBrains Mono\',monospace; font-size:0.62rem; '
                f'color:#5A7FA0; letter-spacing:0.04em;">'
                f'because you liked: {" &nbsp;·&nbsp; ".join(reason_parts)}'
                f'  </span>'
                f'</div>'
            )

        cards_html += (
            f'<div class="rec-card" style="flex-direction:column; align-items:flex-start; padding:0.75rem 0.85rem; margin-bottom:0.55rem;">'
            f'  <div style="display:flex; align-items:center; gap:0.8rem; width:100%;">'
            f'    <span class="rec-rank">#{rank:02d}</span>'
            f'    <span style="font-family:\'Inter\',sans-serif; font-size:0.88rem; font-weight:500; color:#C9D1D9;">{title}</span>'
            f'  </div>'
            f'  <div style="margin-top:0.4rem; padding-left:2.5rem;">{genre_chips}</div>'
            f'  {reason_html}'
            f'</div>'
        )

    st.markdown(cards_html, unsafe_allow_html=True)
