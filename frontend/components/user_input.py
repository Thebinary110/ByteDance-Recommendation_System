"""
User input component — renders the user ID field and load trigger.
"""

import streamlit as st


def render_user_input() -> tuple[int | None, bool]:
    """
    Renders the user section.
    Returns (user_id, load_clicked).
    """
    st.markdown(
        '<p class="section-label">USER SESSION</p>',
        unsafe_allow_html=True,
    )

    col_input, col_btn = st.columns([3, 1], gap="medium")

    with col_input:
        user_id = st.number_input(
            label="User ID",
            min_value=1,
            max_value=99999,
            value=st.session_state.get("user_id", 1),
            step=1,
            label_visibility="collapsed",
            key="user_id_input",
        )

    with col_btn:
        load_clicked = st.button(
            "Load Recommendations",
            use_container_width=True,
            key="load_btn",
        )

    return int(user_id), load_clicked
