"""
Global and specific CSS styles for the Netflix Churn Prediction Dashboard.
Includes theme colors, landing page background, and custom button styles.
"""

import streamlit as st


def apply_global_styles():
    """Applies global Netflix-themed CSS styles across the entire application."""
    st.markdown(
        """
    <style>
        /* Netflix theme colors */
        :root {
            --netflix-red: #E50914;
            --netflix-black: #141414;
            --netflix-dark-gray: #221f1f;
            --netflix-gray: #564d4d;
        }

        /* Main background with Netflix gradient */
        .main {
            background: linear-gradient(to bottom, #141414 0%, #000000 100%);
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #221f1f 0%, #141414 100%);
            padding-top: 0.1rem !important;
        }

        /* Sidebar buttons: keep label horizontal (prevent vertical letter stacking) */
        [data-testid="stSidebar"] .stButton > button {
            white-space: nowrap !important;
            min-width: 4rem;
        }

        /* Sidebar text visibility */
        [data-testid="stSidebar"] * {
            color: white !important;
        }

        /* Sidebar labels */
        [data-testid="stSidebar"] label {
            color: #E50914 !important;
            font-weight: 600;
        }

        /* Radio buttons */
        [data-testid="stSidebar"] [data-baseweb="radio"] {
            color: white !important;
        }

        /* Sidebar radio options: more vertical space between choices */
        [data-testid="stSidebar"] .stRadio > div {
            margin-bottom: 1rem !important;
        }
        [data-testid="stSidebar"] .stRadio [role="radio"] {
            margin-bottom: 0.75rem !important;
        }

        /* Headers */
        h1, h2, h3 {
            color: #E50914 !important;
            font-family: 'Bebas Neue', sans-serif;
            letter-spacing: 2px;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #E50914;
            font-size: 2rem !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: #E50914;
            color: white;
            border-radius: 4px;
            border: none;
            font-weight: bold;
            padding: 0.5rem 2rem;
            transition: all 0.3s;
        }

        .stButton>button:hover {
            background-color: #B20710;
            transform: scale(1.05);
        }

        /* Landing profile buttons: override when #landing-marker exists */
        #landing-marker ~ * .stButton > button {
            width: 420px !important;
            height: 420px !important;
            min-width: 38vmin !important;
            min-height: 38vmin !important;
            max-width: 420px !important;
            max-height: 420px !important;
            padding: 0 !important;
            background: rgba(30, 20, 40, 0.85) !important;
            backdrop-filter: blur(10px) !important;
            border: 3px solid rgba(255, 215, 0, 0.2) !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5) !important;
            margin: 0 auto 0.5rem !important;
            font-size: 6rem !important;
            line-height: 1 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        #landing-marker ~ * .stButton > button:hover {
            border-color: rgba(255, 215, 0, 0.6) !important;
            background: rgba(50, 30, 60, 0.95) !important;
            transform: scale(1.05) !important;
            box-shadow: 0 8px 30px rgba(255, 215, 0, 0.3) !important;
        }
        #landing-marker ~ * .stButton {
            display: flex !important;
            justify-content: center !important;
        }

        /* Dataframes */
        .dataframe {
            background-color: #221f1f !important;
        }

        /* Expander */
        .streamlit-expanderHeader {
            background-color: #221f1f;
            border-radius: 4px;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: #221f1f;
            border-radius: 4px 4px 0 0;
            color: white;
        }

        .stTabs [aria-selected="true"] {
            background-color: #E50914;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def apply_landing_styles():
    """Landing: black background, hide sidebar, column spacing.
    Button size is set in apply_global_styles().
    """
    st.markdown(
        """
    <style>
        [data-testid="stSidebar"] { display: none; }
        .stApp { background: #000000 !important; }
        .main { background: transparent !important; }

        #landing-marker ~ * [data-testid="column"] {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def apply_play_button_styles():
    """Legacy: profile selection is now the square button; kept for any other button styling."""
    st.markdown(
        """
    <style>
        /* Prevent button container from expanding */
        div.stButton {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 2rem;
        }

        /* Button Body: Fixed dimensions from all directions */
        div.stButton > button {
            width: 70px !important;
            height: 70px !important;
            min-width: 70px !important;
            max-width: 70px !important;
            min-height: 70px !important;
            max-height: 70px !important;
            
            border-radius: 50% !important; /* Perfect circle */
            padding: 0 !important;
            margin: 0 auto !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            
            /* Design aesthetics */
            background: rgba(229, 9, 20, 0.95) !important;
            border: 3px solid rgba(255, 215, 0, 0.6) !important;
            font-size: 2rem !important;
            line-height: 1 !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.7) !important;
            transition: all 0.3s ease !important;
        }

        /* Hover effect */
        div.stButton > button:hover {
            background: rgba(255, 215, 0, 0.95) !important;
            border-color: rgba(255, 215, 0, 1) !important;
            transform: scale(1.1) !important;
        }

        /* Prevent text misalignment in browsers like Firefox */
        div.stButton > button p {
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1 !important;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
