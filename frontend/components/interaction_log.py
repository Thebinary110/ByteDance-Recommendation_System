"""
Interaction log component — displays a live table of user actions.
"""

import streamlit as st

from utils.helpers import format_interaction_row


def render_interaction_log() -> None:
    st.markdown(
        '<p class="section-label">INTERACTION LOG</p>',
        unsafe_allow_html=True,
    )

    log: list[dict] = st.session_state.get("interaction_log", [])

    if not log:
        st.markdown(
            '<p class="empty-state">No interactions recorded yet.</p>',
            unsafe_allow_html=True,
        )
        return

    rows = [format_interaction_row(entry) for entry in reversed(log)]
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "User ID": st.column_config.TextColumn("User ID", width="small"),
            "Item ID": st.column_config.TextColumn("Item ID", width="small"),
            "Title": st.column_config.TextColumn("Title", width="large"),
            "Action": st.column_config.TextColumn("Action", width="small"),
            "Timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
        },
    )

    if st.button("Clear Log", key="clear_log_btn"):
        st.session_state.interaction_log = []
        st.rerun()
