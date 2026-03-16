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
    """Applies Bridgerton-inspired elegant CSS for the 'Who's Watching?' landing page."""
    st.markdown(
        """
    <style>
        /* Hide sidebar */
        [data-testid="stSidebar"] {
            display: none;
        }

        /* Bridgerton-inspired elegant background */
        .stApp {
            background:
                repeating-linear-gradient(45deg, transparent, transparent 35px, 
                rgba(139, 69, 19, 0.03) 35px, rgba(139, 69, 19, 0.03) 70px),
                repeating-linear-gradient(-45deg, transparent, transparent 35px, 
                rgba(75, 0, 130, 0.03) 35px, rgba(75, 0, 130, 0.03) 70px),
                radial-gradient(circle at 30% 20%, rgba(75, 0, 130, 0.2) 0%, transparent 50%),
                radial-gradient(circle at 70% 80%, rgba(139, 69, 19, 0.2) 0%, transparent 50%),
                linear-gradient(135deg, #0f0c1d 0%, #1a0e2e 50%, #2d1b3d 100%) !important;
            background-size: cover !important;
            background-attachment: fixed !important;
        }

        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at 50% 50%, 
            rgba(255, 215, 0, 0.05) 0%, transparent 70%);
            z-index: 0;
            pointer-events: none;
        }

        .main {
            background: transparent !important;
            position: relative;
            z-index: 1;
        }

        .stButton > button {
            background: rgba(30, 20, 40, 0.85) !important;
            backdrop-filter: blur(10px) !important;
            border: 3px solid rgba(255, 215, 0, 0.2) !important;
            border-radius: 8px !important;
            min-height: 300px !important;
            transition: all 0.3s !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5) !important;
        }

        .stButton > button:hover {
            border-color: rgba(255, 215, 0, 0.6) !important;
            background: rgba(50, 30, 60, 0.95) !important;
            transform: scale(1.05) !important;
            box-shadow: 0 8px 30px rgba(255, 215, 0, 0.3) !important;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def apply_play_button_styles():
    """Prevents profile play icon buttons from distorting - enforces fixed circular dimensions."""
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
