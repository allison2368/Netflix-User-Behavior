"""
Landing Page for the Netflix Churn Prediction Dashboard.
Displays the 'Who's Watching?' profile selection screen.
"""

import streamlit as st

import styles


def show_landing_page():
    """Renders the Netflix-style profile selection landing page."""
    st.markdown(
        '<div id="landing-marker" style="display:none" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )

    # Large Netflix logo section
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        logo_html = (
            "<div style='text-align: center;'>"
            "<h1 style='font-size: 5rem; font-weight: 900; color: #E50914; "
            'letter-spacing: -0.1rem; font-family: "Bebas Neue", "Impact", '
            "sans-serif; margin: 0; text-shadow: 0 0 20px rgba(229, 9, 20, 0.6), "
            "0 0 40px rgba(229, 9, 20, 0.4), 2px 2px 8px rgba(0, 0, 0, 0.9);'>"
            "NETFLIX</h1></div>"
        )
        st.markdown(logo_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Title section
    title_html = (
        "<h1 style='text-align: center; color: white; font-size: 3.5rem; "
        "font-weight: 400; letter-spacing: 2px; margin-bottom: 3rem; "
        "text-shadow: 0 0 20px rgba(229, 9, 20, 0.5), 0 0 40px "
        "rgba(229, 9, 20, 0.3), 2px 2px 8px rgba(0, 0, 0, 0.9);'>"
        "Who's Watching?</h1>"
    )
    st.markdown(title_html, unsafe_allow_html=True)

    # Profile selection: plain red square above each name; spacer columns center the four users
    _spacer_l, col1, col2, col3, col4, _spacer_r = st.columns([1.5, 1, 1, 1, 1, 1])
    profiles = [
        {"key": "marcus", "name": "Marcus", "role": "Marketing Manager", "col": col1},
        {"key": "sarah", "name": "Sarah", "role": "Content Executive", "col": col2},
        {"key": "puja", "name": "Puja", "role": "Product Manager", "col": col3},
        {"key": "admin", "name": "Admin", "role": "Model Performance", "col": col4},
    ]

    for profile in profiles:
        with profile["col"]:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            if st.button("👤", key=profile["key"]):
                st.session_state.selected_user = profile["key"]
                st.rerun()
            st.markdown(
                f"<div style='color: white; font-size: 1.5rem; margin-top: 0.5rem;'>"
                f"{profile['name']}</div>"
                f"<div style='color: white; font-size: 0.9rem; margin-bottom: 1rem;'>"
                f"{profile['role']}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br><br>", unsafe_allow_html=True)
    # Apply landing styles last so they override any Streamlit defaults and the buttons render large
    styles.apply_landing_styles()
