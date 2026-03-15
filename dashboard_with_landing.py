"""
Netflix Churn Prediction Dashboard
Interactive Streamlit dashboard based on notebook churn_model-2.ipynb
Imports all model functionality from churn_model.py

STORAGE OPTIONS:
  - Local (default):  Loads from ./model_outputs/
  - GCS (optional):   Loads from Google Cloud Storage bucket
  
To use GCS, set USE_GCS = True in the configuration below
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import churn_model as cm

UC2_TEAL         = "#1D9E75"
UC2_AMBER        = "#BA7517"
UC2_ORIGIN_COLORS = {"Netflix Original": UC2_TEAL, "Licensed": UC2_AMBER}

# BigQuery imports for Use Case 3
try:
    from dotenv import load_dotenv
    from google.cloud import bigquery
    load_dotenv()
    BIGQUERY_AVAILABLE = True
    PROJECT_ID = os.getenv("PROJECT_ID")
except ImportError:
    BIGQUERY_AVAILABLE = False
    PROJECT_ID = None


# ============================================================================
# STORAGE CONFIGURATION
# ============================================================================

# Set to True to load from Google Cloud Storage instead of local
USE_GCS = False  

# GCS bucket name (only used if USE_GCS = True)
BUCKET_NAME = 'netflix-churn-models'


# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="Netflix Churn Prediction Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Netflix-themed custom CSS
st.markdown("""
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
""", unsafe_allow_html=True)


# ============================================================================
# LOAD MODEL ARTIFACTS
# ============================================================================

@st.cache_resource
def load_model_artifacts():
    """Load all saved model artifacts using churn_model module"""
    try:
        if USE_GCS:
            artifacts = cm.load_artifacts(
                use_gcs=True,
                bucket_name=BUCKET_NAME
            )
        else:
            artifacts = cm.load_artifacts('./model_outputs')
        return artifacts
    except FileNotFoundError as e:
        st.error("❌ **Model artifacts not found!**")
        if USE_GCS:
            st.info(f"📝 Models not found in GCS bucket: gs://{BUCKET_NAME}/model_outputs/")
            st.info("Please run `train_model.py` with USE_GCS = True first.")
        else:
            st.info("📝 Please run `train_model.py` first to train the model and generate artifacts.")
        st.code("python train_model.py", language="bash")
        st.stop()


# ============================================================================
# USE CASE 3 DATA LOADING FUNCTIONS (PUJA - PRODUCT MANAGER)
# ============================================================================

@st.cache_data
def load_null_search_analysis():
    """
    Analyzes search failure rates (Results returned but no click)
    split by Churn vs. Not Churn groups.
    """
    if not BIGQUERY_AVAILABLE or not PROJECT_ID:
        return pd.DataFrame()
    
    client = bigquery.Client(project=PROJECT_ID)

    # SQL: Calculate failure rate
    sql = """
    WITH user_search_stats AS (
        SELECT 
            s.user_id,
            COUNT(s.Search_id) as total_searches,
            -- Failed search: Results were returned (>0) but the user did not click
            SUM(CASE WHEN s.Clicked = 0 AND s.Results_returned > 0 THEN 1 ELSE 0 END) as failed_searches
        FROM `netflix-user-behavior.kaggle_cleaned.search_logs_cleaned` s
        GROUP BY s.user_id
    ),
    user_labels AS (
        SELECT User_id,
               CASE WHEN watch_decline_ratio < 0.2 THEN 'Churn' ELSE 'Not Churn' END as churn_label
        FROM `netflix-user-behavior.kaggle_cleaned.churn_features`
    )
    SELECT 
        l.churn_label,
        -- Calculate average search failure rate (%) per user
        AVG(CAST(s.failed_searches AS FLOAT64) / NULLIF(s.total_searches, 0)) * 100 as avg_search_failure_rate
    FROM user_labels l
    JOIN user_search_stats s ON l.User_id = s.user_id
    GROUP BY 1
    """
    return client.query(sql).to_dataframe()


@st.cache_data
def load_failed_queries():
    """Fetches the top 10 search queries that led to no clicks for the Churn group."""
    if not BIGQUERY_AVAILABLE or not PROJECT_ID:
        return pd.DataFrame()
    
    client = bigquery.Client(project=PROJECT_ID)

    sql = """
    WITH user_labels AS (
        SELECT User_id,
               CASE WHEN watch_decline_ratio < 0.2 THEN 'Churn' ELSE 'Not Churn' END as churn_label
        FROM `netflix-user-behavior.kaggle_cleaned.churn_features`
    )
    SELECT 
        s.search_query,
        COUNT(*) as failure_count
    FROM `netflix-user-behavior.kaggle_cleaned.search_logs_cleaned` s
    JOIN user_labels l ON s.user_id = l.User_id
    WHERE l.churn_label = 'Churn' 
      AND s.Clicked = 0 
      AND s.Results_returned > 0
    GROUP BY 1
    ORDER BY failure_count DESC
    LIMIT 10
    """
    return client.query(sql).to_dataframe()


@st.cache_data
def load_tenure_analysis():
    """Groups churn rates by user tenure in months."""
    if not BIGQUERY_AVAILABLE or not PROJECT_ID:
        return pd.DataFrame()
    
    client = bigquery.Client(project=PROJECT_ID)

    # Convert tenure_days into Months (30-day buckets)
    sql = """
    WITH user_tenure AS (
        SELECT 
            User_id,
            -- Floor division to group days into months
            FLOOR(tenure_days / 30) as tenure_months,
            CASE WHEN watch_decline_ratio < 0.2 THEN 1 ELSE 0 END as is_churn
        FROM `netflix-user-behavior.kaggle_cleaned.churn_features`
    )
    SELECT 
        tenure_months,
        COUNT(*) as total_users,
        AVG(is_churn) * 100 as churn_rate
    FROM user_tenure
    GROUP BY 1
    ORDER BY 1
    """
    return client.query(sql).to_dataframe()


def plot_tenure(df):
    """Generates a line plot showing churn rate trends by subscription tenure."""
    fig, ax = plt.subplots(figsize=(12, 6))

    if df is None or df.empty or "tenure_months" not in df.columns:
        ax.text(0.5, 0.5, "No Data Available", ha="center")
        return fig

    sns.lineplot(
        data=df,
        x="tenure_months",
        y="churn_rate",
        marker="o",
        color="red",
        linewidth=2,
        ax=ax,
    )

    # Add a baseline for the overall average churn rate (approx. 8%)
    ax.axhline(8.0, color="gray", linestyle="--", label="Average Churn Rate (8%)")

    ax.set_title("Is Churn higher for Newbies or Veterans?", fontsize=15, pad=20)
    ax.set_xlabel("Tenure (Months)", fontsize=12)
    ax.set_ylabel("Churn Rate (%)", fontsize=12)

    # Make sure not accessing empty data
    if not df.empty:
        ax.set_ylim(0, max(df["churn_rate"]) + 5)

    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


@st.cache_data
def load_final_pm_report():
    """Calculates Bounce Rate per Genre for Churn vs. Not Churn groups."""
    if not BIGQUERY_AVAILABLE or not PROJECT_ID:
        return pd.DataFrame()
    
    client = bigquery.Client(project=PROJECT_ID)

    sql = """
    SELECT 
        m.genre_primary,
        CASE WHEN f.watch_decline_ratio < 0.2 THEN 'Churn' ELSE 'Not Churn' END as churn_label,
        COUNT(*) as total_views,
        AVG(CASE WHEN w.watch_duration_minutes < 5 THEN 1 ELSE 0 END) as bounce_rate
    FROM `netflix-user-behavior.kaggle_cleaned.watch_history_cleaned` w
    JOIN `netflix-user-behavior.kaggle_cleaned.movies_cleaned` m ON w.movie_id = m.movie_id
    JOIN `netflix-user-behavior.kaggle_cleaned.churn_features` f ON w.user_id = f.user_id
    GROUP BY 1, 2
    """
    return client.query(sql).to_dataframe()


def plot_pm_report(df):
    """Generates a bar plot comparing bounce rates by genre across groups."""
    plt.style.use("dark_background")  # dark theme
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=df,
        x="bounce_rate",
        y="genre_primary",
        hue="churn_label",
        palette="magma",
        ax=ax,
    )

    ax.set_title("Which Genre Disappoints Users the Most? (Bounce Rate)", fontsize=15)
    ax.set_xlabel("Bounce Rate (Watched < 5 mins)")
    ax.set_ylabel("Genre")

    # Add vertical line for overall average bounce rate
    ax.axvline(
        df["bounce_rate"].mean(), color="red", linestyle="--", label="Overall Avg"
    )
    ax.legend()

    return fig


def plot_feature_importance_from_csv():
    """
    Plots feature importance using the pre-saved CSV file.
    """
    name_mapping = {
        "engagement_ratio_7v30": "Engagement Drop (Last 7d vs 30d)",
        "days_since_last_watch": "Days Since Last View",
        "watch_last_30d": "Viewing Time (Last 30 Days)",
        "total_sessions": "Total App Sessions",
        "avg_completion_rate": "Content Completion Rate (%)",
        "segment": "User Segment Group",
        "Monthly Subscription Plan Amount": "Monthly Bill Amount",
        "avg_rec_score_seen": "Recommendation Relevance Score",
        "Subscription Tenure": "Total Membership Days",
        "Average Search Time": "Time Spent Searching",
    }
    df_importance = pd.read_csv("./model_outputs/feature_importance.csv")

    # Change variables more friendly format
    df_importance["feature"] = df_importance["feature"].replace(name_mapping)

    # Sort by importance
    df_importance = df_importance.sort_values(by="importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df_importance,
        x="importance",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
        ax=ax,
    )

    ax.set_title("Top 10 Drivers of User Churn", fontsize=15)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")

    # Show the importance numerically
    for i, v in enumerate(df_importance["importance"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontweight="bold")
    return fig


def get_summary_metrics():
    """
    Aggregates main KPIs for the dashboard overview.
    Calls cached functions for high-performance data retrieval.
    """
    df_bounce = load_final_pm_report()
    df_search = load_null_search_analysis()
    df_tenure = load_tenure_analysis()

    avg_bounce = df_bounce["bounce_rate"].mean() * 100 if not df_bounce.empty else 0
    avg_search = df_search["avg_search_failure_rate"].mean() if not df_search.empty else 0
    overall_churn = df_tenure["churn_rate"].mean() if not df_tenure.empty else 0

    return overall_churn, avg_bounce, avg_search


# ============================================================================
# LANDING PAGE - WHO'S WATCHING?
# ============================================================================

def show_landing_page():
    """Netflix-style 'Who's Watching?' landing page with Bridgerton-inspired elegant background"""
    
    # Bridgerton-inspired elegant background using CSS only (no external images)
    st.markdown("""
    <style>
        /* Hide sidebar */
        [data-testid="stSidebar"] {
            display: none;
        }
        
        /* Bridgerton-inspired elegant background using CSS only */
        .stApp {
            background: 
                /* Royal purple and gold Victorian pattern */
                repeating-linear-gradient(45deg, transparent, transparent 35px, rgba(139, 69, 19, 0.03) 35px, rgba(139, 69, 19, 0.03) 70px),
                repeating-linear-gradient(-45deg, transparent, transparent 35px, rgba(75, 0, 130, 0.03) 35px, rgba(75, 0, 130, 0.03) 70px),
                /* Rich gradient base */
                radial-gradient(circle at 30% 20%, rgba(75, 0, 130, 0.2) 0%, transparent 50%),
                radial-gradient(circle at 70% 80%, rgba(139, 69, 19, 0.2) 0%, transparent 50%),
                linear-gradient(135deg, #0f0c1d 0%, #1a0e2e 50%, #2d1b3d 100%) !important;
            background-size: cover !important;
            background-attachment: fixed !important;
        }
        
        /* Elegant golden glow overlay */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at 50% 50%, rgba(255, 215, 0, 0.05) 0%, transparent 70%);
            z-index: 0;
            pointer-events: none;
        }
        
        /* Main content */
        .main {
            background: transparent !important;
            position: relative;
            z-index: 1;
        }
        
        /* Profile cards with elegant Victorian styling */
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
    """, unsafe_allow_html=True)
    
    # Large Netflix logo
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="font-size: 4rem; font-weight: 900; color: #E50914; letter-spacing: -0.1rem; 
                       font-family: 'Bebas Neue', 'Impact', sans-serif; margin: 0;
                       text-shadow: 0 0 20px rgba(229, 9, 20, 0.6), 0 0 40px rgba(229, 9, 20, 0.4), 2px 2px 8px rgba(0, 0, 0, 0.9);">
                NETFLIX
            </h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Title
    st.markdown("""
    <h1 style="text-align: center; color: white; font-size: 3.5rem; font-weight: 400; letter-spacing: 2px; margin-bottom: 3rem;
               text-shadow: 0 0 20px rgba(229, 9, 20, 0.5), 0 0 40px rgba(229, 9, 20, 0.3), 2px 2px 8px rgba(0, 0, 0, 0.9);">
        Who's Watching?
    </h1>
    """, unsafe_allow_html=True)
    
    # Profile grid with Netflix-style cartoon avatars
    col1, col2, col3, col4 = st.columns(4)
    
    profiles = [
        {
            "key": "marcus",
            "emoji": "🎯",
            "name": "Marcus",
            "role": "Marketing Manager",
            "col": col1,
            "color": "linear-gradient(135deg, #E50914 0%, #B20710 100%)"
        },
        {
            "key": "sarah",
            "emoji": "📺",
            "name": "Sarah",
            "role": "Content Executive",
            "col": col2,
            "color": "linear-gradient(135deg, #564d4d 0%, #221f1f 100%)"
        },
        {
            "key": "puja",
            "emoji": "💡",
            "name": "Puja",
            "role": "Product Manager",
            "col": col3,
            "color": "linear-gradient(135deg, #FFD700 0%, #FFA500 100%)"
        },
        {
            "key": "admin",
            "emoji": "📊",
            "name": "Admin",
            "role": "Model Performance",
            "col": col4,
            "color": "linear-gradient(135deg, #1E90FF 0%, #0066CC 100%)"
        }
    ]
    
    for profile in profiles:
        with profile["col"]:
            # Display avatar and info
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <div style="width: 140px; height: 140px; margin: 0 auto 1rem; border-radius: 8px;
                            background: {profile['color']}; 
                            display: flex; align-items: center; justify-content: center;
                            font-size: 5rem; border: 4px solid rgba(255,215,0,0.3);
                            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
                            transition: all 0.3s ease;">
                    {profile['emoji']}
                </div>
                <div style="color: #808080; font-size: 1.5rem; text-shadow: 1px 1px 4px black; margin-top: 0.5rem;">{profile['name']}</div>
                <div style="color: #565656; font-size: 0.9rem; margin-bottom: 1rem;">{profile['role']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Centered button below
            col_a, col_b, col_c = st.columns([1.5, 1, 1.5])
            with col_b:
                if st.button(
                    "▶️",
                    key=profile['key'], 
                    help=f"Click to view {profile['name']}'s dashboard"
                ):
                    st.session_state.selected_user = profile['key']
                    st.rerun()
    
    # Style the icon buttons
    st.markdown("""
    <style>
        /* Style the play icon buttons */
        .stButton > button {
            font-size: 2rem !important;
            padding: 0.6rem !important;
            border-radius: 50% !important;
            background: rgba(229, 9, 20, 0.95) !important;
            border: 3px solid rgba(255, 215, 0, 0.6) !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.7) !important;
            width: 60px !important;
            height: 60px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            cursor: pointer !important;
        }
        
        .stButton > button:hover {
            background: rgba(255, 215, 0, 0.95) !important;
            border-color: rgba(255, 215, 0, 1) !important;
            transform: scale(1.15) !important;
            box-shadow: 0 6px 30px rgba(255, 215, 0, 0.6) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)




# ============================================================================
# MAIN DASHBOARD LAYOUT
# ============================================================================

def main():
    # Initialize session state for user selection
    if 'selected_user' not in st.session_state:
        st.session_state.selected_user = None
    
    # Show landing page or dashboard
    if st.session_state.selected_user is None:
        show_landing_page()
    else:
        # Load artifacts
        artifacts = load_model_artifacts()
        
        # Header with Back button and Netflix branding
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("⬅ Switch User", help="Return to profile selection"):
                st.session_state.selected_user = None
                st.rerun()
        
        with col2:
            st.title("🎬 NETFLIX CHURN PREDICTION DASHBOARD")
            st.markdown("*Powered by Random Forest Machine Learning*")
        
        with col3:
            st.markdown("""
            <div style="text-align: right;">
                <h2 style="font-size: 2rem; font-weight: 900; color: #E50914; letter-spacing: -0.05rem; 
                           font-family: 'Bebas Neue', 'Impact', sans-serif; margin: 0;">
                    NETFLIX
                </h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Sidebar Navigation
        st.sidebar.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2 style="font-size: 2rem; font-weight: 900; color: #E50914; letter-spacing: -0.05rem; 
                       font-family: 'Bebas Neue', 'Impact', sans-serif; margin: 0;">
                NETFLIX
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Show current user info
        user_info = {
            "marcus": {"emoji": "🎯", "name": "Marcus", "role": "Marketing Manager"},
            "sarah": {"emoji": "📺", "name": "Sarah", "role": "Content Executive"},
            "puja": {"emoji": "💡", "name": "Puja", "role": "Product Manager"},
            "admin": {"emoji": "📊", "name": "Admin", "role": "ML Engineer"}
        }
        
        current_user = user_info.get(st.session_state.selected_user, {})
        st.sidebar.success(f"""
        **Current User:**
        
        {current_user.get('emoji', '👤')} **{current_user.get('name', 'User')}**
        
        *{current_user.get('role', 'Role')}*
        """)
        
        st.sidebar.markdown("---")
        st.sidebar.title("📍 Navigation")
        
        # Default page based on user
        user_default_pages = {
            "marcus": "🎯 Marketing Campaign (Marcus)",
            "sarah": "📺 Content Investment (Sarah)",
            "puja": "💡 Feature Engagement (Puja)",
            "admin": "📊 Model Performance"
        }
        
        default_page = user_default_pages.get(st.session_state.selected_user, "🎯 Marketing Campaign (Marcus)")
        
        page = st.sidebar.radio(
            "Select Dashboard View",
            [
                "🎯 Marketing Campaign (Marcus)",
                "📺 Content Investment (Sarah)", 
                "💡 Feature Engagement (Puja)",
                "📊 Model Performance"
            ],
            index=["🎯 Marketing Campaign (Marcus)", 
                   "📺 Content Investment (Sarah)", 
                   "💡 Feature Engagement (Puja)", 
                   "📊 Model Performance"].index(default_page),
            help="Choose the user persona and their specific use case"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🤖 Model Information")
        st.sidebar.metric("Model Type", "Random Forest")
        st.sidebar.metric("Trees", artifacts['config'].get('RF_N_ESTIMATORS', 1000))
        st.sidebar.metric("Threshold", f"{artifacts['config']['PREDICTION_THRESHOLD']}")
        st.sidebar.metric("Churn Definition", f"< {artifacts['config']['CHURN_THRESHOLD']} watch decline")
        
        # Route to pages based on selection
        if page == "🎯 Marketing Campaign (Marcus)":
            show_marketing_campaign(artifacts)
        elif page == "📺 Content Investment (Sarah)":
            show_content_investment(artifacts)
        elif page == "💡 Feature Engagement (Puja)":
            show_feature_engagement(artifacts)
        elif page == "📊 Model Performance":
            show_model_performance(artifacts)


# ============================================================================
# USE CASE 1: MARKETING CAMPAIGN (MARCUS)
# ============================================================================

def show_marketing_campaign(artifacts):
    st.header("🎯 USE CASE 1: Targeted Marketing Campaigns")
    
    st.markdown("""
    **Actor:** Marcus, Marketing Manager  
    **Goal:** Identify high-risk churners and export for discount campaigns  
    **Action:** Filter by churn probability → Calculate revenue at risk → Export segment
    """)
    
    st.markdown("---")
    
    # STEP 1: Set Filters
    st.subheader("STEP 1: Set Risk Criteria")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        churn_threshold = st.slider(
            "🎯 Churn Probability Threshold (%)",
            min_value=50,
            max_value=100,
            value=85,
            step=5,
            help="Users with churn probability above this threshold will be flagged as high-risk"
        )
    
    with col2:
        n_segments = artifacts['config']['N_CLUSTERS']
        segment_filter = st.multiselect(
            "📊 Customer Segments",
            options=[f"Segment {i}" for i in range(n_segments)],
            default=[f"Segment {i}" for i in range(n_segments)],
            help="Filter specific customer behavioral segments"
        )
    
    with col3:
        min_revenue = st.number_input(
            "💰 Min Monthly Revenue ($)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=5.0,
            help="Minimum subscription fee to target high-value customers"
        )
    
    # Generate sample high-risk users (in production, query from BigQuery)
    np.random.seed(42)
    n_high_risk = 350
    
    high_risk_users = pd.DataFrame({
        'user_id': [f'USER_{i:05d}' for i in range(1, n_high_risk + 1)],
        'churn_probability': np.random.uniform(0.5, 1.0, n_high_risk),
        'segment': np.random.choice([f"Segment {i}" for i in range(n_segments)], n_high_risk),
        'days_since_last_watch': np.random.randint(10, 90, n_high_risk),
        'total_sessions': np.random.randint(3, 80, n_high_risk),
        'avg_completion_rate': np.random.uniform(0.15, 0.85, n_high_risk),
        'monthly_revenue': np.random.choice([9.99, 15.49, 19.99], n_high_risk),
        'subscription_plan': np.random.choice(['Basic', 'Standard', 'Premium'], n_high_risk),
        'lifetime_value': np.random.uniform(50, 800, n_high_risk),
        'tenure_months': np.random.randint(1, 60, n_high_risk)
    })
    
    # Apply filters
    filtered_users = high_risk_users[
        (high_risk_users['churn_probability'] >= churn_threshold/100) &
        (high_risk_users['segment'].isin(segment_filter)) &
        (high_risk_users['monthly_revenue'] >= min_revenue)
    ].copy()
    
    # Add risk categorization
    filtered_users['risk_level'] = pd.cut(
        filtered_users['churn_probability'],
        bins=[0, 0.7, 0.85, 1.0],
        labels=['Medium Risk', 'High Risk', 'Critical Risk']
    )
    
    # STEP 2: Calculate Revenue at Risk
    st.markdown("---")
    st.subheader("STEP 2: Revenue Impact Analysis")
    
    total_revenue_at_risk = filtered_users['monthly_revenue'].sum()
    annual_revenue_at_risk = total_revenue_at_risk * 12
    avg_churn_prob = filtered_users['churn_probability'].mean()
    total_lifetime_value_at_risk = filtered_users['lifetime_value'].sum()
    expected_churn_loss = total_revenue_at_risk * avg_churn_prob
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🚨 High-Risk Users",
            f"{len(filtered_users):,}",
            delta=f"{len(filtered_users)/len(high_risk_users)*100:.1f}% of total",
            help="Number of users matching your risk criteria"
        )
    
    with col2:
        st.metric(
            "💵 Monthly Revenue at Risk",
            f"${total_revenue_at_risk:,.2f}",
            delta=f"${annual_revenue_at_risk:,.0f} annually",
            delta_color="inverse",
            help="Total MRR from high-risk segment"
        )
    
    with col3:
        st.metric(
            "📉 Avg Churn Probability",
            f"{avg_churn_prob:.1%}",
            help="Average likelihood of churn in this segment"
        )
    
    with col4:
        st.metric(
            "💎 Lifetime Value at Risk",
            f"${total_lifetime_value_at_risk:,.0f}",
            help="Total estimated customer lifetime value at risk"
        )
    
    with col5:
        st.metric(
            "⚠️ Expected Loss",
            f"${expected_churn_loss:,.2f}",
            delta=f"Monthly",
            delta_color="inverse",
            help="Revenue at risk × Avg churn probability"
        )
    
    # STEP 3: Segment Analysis
    st.markdown("---")
    st.subheader("STEP 3: Segment Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution by segment
        segment_stats = filtered_users.groupby('segment').agg({
            'user_id': 'count',
            'monthly_revenue': 'sum',
            'churn_probability': 'mean',
            'lifetime_value': 'sum'
        }).reset_index()
        segment_stats.columns = ['Segment', 'Users', 'MRR at Risk', 'Avg Churn Prob', 'LTV at Risk']
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=segment_stats['Segment'],
                y=segment_stats['Users'],
                name='User Count',
                marker_color='#E50914',
                text=segment_stats['Users'],
                textposition='outside'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=segment_stats['Segment'],
                y=segment_stats['Avg Churn Prob'] * 100,
                name='Avg Churn %',
                marker_color='#ffffff',
                mode='lines+markers',
                line=dict(width=3)
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title="High-Risk Users by Customer Segment",
            xaxis_title="Customer Segment",
            template="plotly_dark",
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_yaxes(title_text="Number of Users", secondary_y=False)
        fig.update_yaxes(title_text="Avg Churn Probability (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue by subscription plan
        plan_revenue = filtered_users.groupby('subscription_plan').agg({
            'monthly_revenue': 'sum',
            'user_id': 'count'
        }).reset_index()
        plan_revenue.columns = ['Plan', 'Revenue', 'Users']
        
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=plan_revenue['Plan'],
            values=plan_revenue['Revenue'],
            marker_colors=['#E50914', '#B20710', '#831010'],
            hole=0.4,
            textinfo='label+percent+value',
            texttemplate='<b>%{label}</b><br>%{percent}<br>$%{value:.2f}',
            hovertemplate='<b>%{label}</b><br>Revenue: $%{value:.2f}<br>Percentage: %{percent}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Monthly Revenue at Risk by Plan",
            template="plotly_dark",
            height=400,
            annotations=[dict(text=f'${total_revenue_at_risk:.0f}', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk level distribution
    col1, col2 = st.columns(2)
    
    with col1:
        risk_dist = filtered_users['risk_level'].value_counts().reset_index()
        risk_dist.columns = ['Risk Level', 'Count']
        
        colors_map = {'Medium Risk': '#FFA500', 'High Risk': '#E50914', 'Critical Risk': '#8B0000'}
        
        fig = go.Figure(data=[go.Bar(
            x=risk_dist['Risk Level'],
            y=risk_dist['Count'],
            marker_color=[colors_map.get(x, '#E50914') for x in risk_dist['Risk Level']],
            text=risk_dist['Count'],
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Distribution by Risk Level",
            xaxis_title="Risk Category",
            yaxis_title="Number of Users",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Churn probability histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=filtered_users['churn_probability'] * 100,
            nbinsx=20,
            marker_color='#E50914',
            opacity=0.8
        ))
        
        fig.add_vline(
            x=churn_threshold,
            line_dash="dash",
            line_color="white",
            annotation_text=f"Threshold: {churn_threshold}%",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title="Churn Probability Distribution",
            xaxis_title="Churn Probability (%)",
            yaxis_title="Number of Users",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # STEP 4: Export Segment
    st.markdown("---")
    st.subheader("STEP 4: Export High-Risk Segment")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Preview of Export Data")
        
        export_df = filtered_users[[
            'user_id', 'segment', 'churn_probability', 'risk_level',
            'subscription_plan', 'monthly_revenue', 'lifetime_value',
            'days_since_last_watch', 'total_sessions', 'avg_completion_rate',
            'tenure_months'
        ]].copy()
        
        export_df['churn_probability'] = (export_df['churn_probability'] * 100).round(1)
        export_df['avg_completion_rate'] = (export_df['avg_completion_rate'] * 100).round(1)
        
        # Create styled dataframe
        styled_df = export_df.head(10).style
        
        # Apply gradient first (needs numeric values)
        styled_df = styled_df.background_gradient(subset=['churn_probability'], cmap='Reds')
        
        # Then apply formatting
        styled_df = styled_df.format({
            'churn_probability': '{:.1f}%',
            'monthly_revenue': '${:.2f}',
            'lifetime_value': '${:.2f}',
            'avg_completion_rate': '{:.1f}%'
        })
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400
        )
        
        st.caption(f"Showing 10 of {len(export_df):,} high-risk users")
    
    with col2:
        st.markdown("### 📤 Export Options")
        
        # Campaign name
        campaign_name = st.text_input(
            "Campaign Name",
            value=f"Churn_Campaign_{pd.Timestamp.now().strftime('%Y%m%d')}",
            help="Name for the marketing campaign"
        )
        
        # Include columns selection
        include_ltv = st.checkbox("Include Lifetime Value", value=True)
        include_behavior = st.checkbox("Include Behavioral Metrics", value=True)
        
        # Prepare export
        export_columns = ['user_id', 'segment', 'churn_probability', 'risk_level', 
                         'subscription_plan', 'monthly_revenue', 'tenure_months']
        
        if include_ltv:
            export_columns.append('lifetime_value')
        
        if include_behavior:
            export_columns.extend(['days_since_last_watch', 'total_sessions', 'avg_completion_rate'])
        
        export_final = export_df[export_columns].copy()
        
        # Export button
        csv = export_final.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{campaign_name}.csv",
            mime="text/csv",
            help="Download the filtered segment for your marketing automation platform",
            use_container_width=True
        )
        
        st.success(f"✅ Ready to export {len(export_final):,} users")
        
        # Campaign summary
        st.markdown("---")
        st.markdown("### 📊 Campaign Summary")
        st.info(f"""
        **Target:** {len(export_final):,} high-risk users  
        **MRR at Risk:** ${total_revenue_at_risk:,.2f}  
        **Avg Churn:** {avg_churn_prob:.1%}  
        **Expected Loss:** ${expected_churn_loss:,.2f}/month  
        
        **Recommended Action:**  
        - 20% discount for 3 months  
        - Estimated retention uplift: 30-40%  
        - ROI: Positive if >25% retained
        """)


# ============================================================================
# USE CASE 2: CONTENT INVESTMENT (SARAH) 
# ============================================================================
@st.cache_data(show_spinner="Loading content data from BigQuery…")
def load_uc2_data() -> tuple:
    """Load session-level and title-level data for Use Case 2.
 
    Returns:
        df      : session-level joined table (watch + movies + churn)
        title_df: title-level aggregated yield table
    """
    client = bigquery.Client(project=PROJECT_ID)
 
    session_sql = """
        SELECT
            w.session_id,
            w.user_id,
            w.movie_id,
            w.watch_date,
            w.watch_duration_minutes,
            w.progress_percentage,
            m.title,
            m.imdb_rating,
            m.genre_primary,
            m.content_type,
            m.is_netflix_original,
            m.is_series,
            m.release_year,
            c.is_active,
            c.watch_decline_ratio,
            c.engagement_ratio_7v30,
            c.watch_last_7d,
            c.watch_last_30d,
            c.monthly_spend,
            c.tenure_days,
            c.subscription_plan,
            c.avg_completion_rate   AS lifetime_completion
        FROM `netflix-user-behavior.kaggle_cleaned.watch_history_cleaned` w
        JOIN `netflix-user-behavior.kaggle_cleaned.movies_cleaned`        m  USING (movie_id)
        JOIN `netflix-user-behavior.kaggle_cleaned.churn_features`        c  USING (user_id)
        WHERE m.imdb_rating IS NOT NULL
    """
 
    title_sql = """
        SELECT
            w.movie_id,
            m.title,
            m.imdb_rating,
            m.genre_primary,
            m.content_type,
            m.is_netflix_original,
            m.is_series,
            COUNT(w.session_id)              AS total_sessions,
            SUM(w.watch_duration_minutes)    AS total_watch_minutes,
            AVG(w.watch_duration_minutes)    AS avg_watch_duration,
            AVG(w.progress_percentage)       AS avg_completion,
            COUNT(DISTINCT w.user_id)        AS unique_viewers
        FROM `netflix-user-behavior.kaggle_cleaned.watch_history_cleaned` w
        JOIN `netflix-user-behavior.kaggle_cleaned.movies_cleaned`        m USING (movie_id)
        WHERE m.imdb_rating IS NOT NULL
        GROUP BY 1,2,3,4,5,6,7
    """
 
    df       = client.query(session_sql).result().to_dataframe()
    title_df = client.query(title_sql).result().to_dataframe()
    return df, title_df
 
 
def _uc2_preprocess(df: pd.DataFrame, title_df: pd.DataFrame):
    """Add derived columns (IMDb bucket, origin label) for Use Case 2.
 
    Args:
        df (pd.DataFrame): Session-level dataframe.
        title_df (pd.DataFrame): Title-level dataframe.
 
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Updated dataframes.
    """
    imdb_bins   = [0, 5, 6, 7, 8, 9, 10]
    imdb_labels = ["0-5", "5-6", "6-7", "7-8", "8-9", "9-10"]
 
    for frame in [df, title_df]:
        frame["imdb_bucket"] = pd.cut(
            frame["imdb_rating"], bins=imdb_bins,
            labels=imdb_labels, right=True, include_lowest=True,
        )
        frame["origin_label"] = np.where(
            frame["is_netflix_original"], "Netflix Original", "Licensed"
        )
 
    return df, title_df
 
 
def _uc2_apply_filters(
    df: pd.DataFrame,
    sel_genres: list,
    sel_plans: list,
    origin: str,
) -> pd.DataFrame:
    """Apply inline filters to the session-level dataframe.
 
    Args:
        df (pd.DataFrame): Session-level dataframe.
        sel_genres (list): Selected genre values.
        sel_plans (list): Selected subscription plan values.
        origin (str): Selected content origin ("All", "Netflix Original", "Licensed").
 
    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    mask = (
        df["genre_primary"].isin(sel_genres)
        & df["subscription_plan"].isin(sel_plans)
    )
    if origin != "All":
        mask &= df["origin_label"] == origin
    return df[mask].copy()
 
 
def _uc2_render_kpis(df: pd.DataFrame, title_df: pd.DataFrame) -> None:
    """Render top-level KPIs for the content portfolio.
 
    Args:
        df (pd.DataFrame): Session-level dataframe.
        title_df (pd.DataFrame): Title-level dataframe.
    """
    st.subheader("📊 Portfolio Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    n_orig     = title_df["is_netflix_original"].sum()
    n_licensed = (~title_df["is_netflix_original"]).sum()
    c1.metric("Total titles",      f"{len(title_df):,}")
    c2.metric("Netflix Originals", f"{n_orig:,}")
    c3.metric("Licensed titles",   f"{n_licensed:,}")
    c4.metric(
        "Avg IMDb (Originals)",
        f"{title_df[title_df.is_netflix_original]['imdb_rating'].mean():.2f}",
    )
    c5.metric(
        "Avg IMDb (Licensed)",
        f"{title_df[~title_df.is_netflix_original]['imdb_rating'].mean():.2f}",
    )
    c6.metric("Active subscriber %", f"{df['is_active'].mean() * 100:.1f}%")
 
 
def _uc2_render_quality_origin(df: pd.DataFrame) -> None:
    """Render quality tier × origin type analysis (completion and watch duration).
 
    Args:
        df (pd.DataFrame): Session-level dataframe.
    """
    import matplotlib.ticker as mticker
 
    st.subheader("1️⃣ Quality Tier × Origin Type → Retention")
    st.caption(
        "Does high-rated content drive more engagement, "
        "and does origin type amplify that effect?"
    )
 
    agg = (
        df.groupby(["origin_label", "imdb_bucket"], observed=True)
        .agg(
            avg_completion=("progress_percentage",    "mean"),
            avg_duration  =("watch_duration_minutes", "mean"),
            sessions      =("session_id",             "count"),
        )
        .reset_index()
    )
 
    col1, col2 = st.columns(2)
 
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for origin, color in UC2_ORIGIN_COLORS.items():
            sub = agg[agg["origin_label"] == origin]
            ax.plot(sub["imdb_bucket"].astype(str), sub["avg_completion"],
                    marker="o", linewidth=2.5, markersize=7,
                    color=color, label=origin)
        ax.set_xlabel("IMDb Rating Bucket")
        ax.set_ylabel("Avg Completion Rate (%)")
        ax.set_title("Completion Rate by Quality Tier & Origin",
                     fontsize=11, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)
 
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        for origin, color in UC2_ORIGIN_COLORS.items():
            sub = agg[agg["origin_label"] == origin]
            ax.plot(sub["imdb_bucket"].astype(str), sub["avg_duration"],
                    marker="o", linewidth=2.5, markersize=7,
                    color=color, label=origin)
        ax.set_xlabel("IMDb Rating Bucket")
        ax.set_ylabel("Avg Watch Duration (min)")
        ax.set_title("Watch Duration by Quality Tier & Origin",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)
 
    pivot = agg.pivot_table(
        index="imdb_bucket", columns="origin_label",
        values="avg_completion", aggfunc="mean",
    ).round(1)
 
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f", cmap="YlGnBu",
                linewidths=0.4, cbar_kws={"label": "Avg Completion %"})
    ax.set_title("Completion % Heatmap — Quality × Origin",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("IMDb Bucket")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close(fig)
 
    with st.expander("📋 Full table"):
        st.dataframe(agg.round(2), use_container_width=True)
 
 
def _uc2_render_kpi_cards(health: pd.DataFrame) -> None:
    """Render subscriber health KPI cards for each origin.
 
    Args:
        health (pd.DataFrame): Aggregated health metrics by origin label.
    """
    for _, row in health.iterrows():
        color = UC2_TEAL if row["origin_label"] == "Netflix Original" else UC2_AMBER
        st.markdown(
            f"<h4 style='color:{color}'>{row['origin_label']}</h4>",
            unsafe_allow_html=True,
        )
        cols = st.columns(6)
        cols[0].metric("Active rate",    f"{row['active_rate']*100:.1f}%")
        cols[1].metric("Avg spend/mo",   f"${row['avg_monthly_spend']:.2f}")
        cols[2].metric("Avg tenure",     f"{row['avg_tenure_days']:.0f}d")
        cols[3].metric("Watch decline",  f"{row['avg_watch_decline']:.2f}")
        cols[4].metric("7v30 ratio",     f"{row['avg_engagement_ratio']:.2f}")
        cols[5].metric("Watch last 30d", f"{row['avg_watch_last30']:.0f} min")
 
 
def _uc2_plot_health_bar(health: pd.DataFrame) -> None:
    """Render bar chart comparing subscriber health metrics by origin.
 
    Args:
        health (pd.DataFrame): Aggregated health metrics by origin label.
    """
    metrics = ["active_rate", "avg_engagement_ratio", "avg_watch_decline"]
    labels  = ["Active Rate", "Engagement 7v30", "Watch Decline Ratio"]
 
    fig, ax = plt.subplots(figsize=(8, 4))
    x     = np.arange(len(metrics))
    width = 0.35
 
    for i, (origin, color) in enumerate(UC2_ORIGIN_COLORS.items()):
        subset = health[health["origin_label"] == origin]
        if subset.empty:
            continue
        heights = [subset[m].values[0] for m in metrics]
        ax.bar(x + i * width, heights, width=width,
               color=color, alpha=0.85, label=origin)
        for xi, h in zip(x + i * width, heights):
            ax.text(xi, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9)
 
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.set_title("Subscriber Health — High-Rated Content Viewers (IMDb ≥ 7)")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)
 
 
def _uc2_plot_active_rate_by_bucket(df: pd.DataFrame) -> None:
    """Render active subscriber rate by IMDb bucket and origin.
 
    Args:
        df (pd.DataFrame): Session-level dataframe.
    """
    import matplotlib.ticker as mticker
 
    high_rated = df[df["imdb_rating"] >= 7].copy()
    active_agg = (
        high_rated.groupby(["origin_label", "imdb_bucket"], observed=True)
        .agg(active_rate=("is_active", "mean"))
        .reset_index()
    )
 
    fig, ax = plt.subplots(figsize=(7, 4))
    for origin, color in UC2_ORIGIN_COLORS.items():
        sub = active_agg[active_agg["origin_label"] == origin]
        ax.plot(
            sub["imdb_bucket"].astype(str),
            sub["active_rate"] * 100,
            marker="o", linewidth=2.5, markersize=7,
            color=color, label=origin,
        )
    ax.set_xlabel("IMDb Rating Bucket")
    ax.set_ylabel("Active Subscriber Rate (%)")
    ax.set_title("Active Rate by Quality Tier & Origin",
                 fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)
 
 
def _uc2_render_subscriber_health(df: pd.DataFrame) -> None:
    """Render subscriber health analysis for high-rated content viewers.
 
    Args:
        df (pd.DataFrame): Session-level dataframe.
    """
    st.subheader("2️⃣ High-Rated Originals vs Licensed → Subscriber Health")
    st.caption(
        "Among users who watch high-rated content (IMDb ≥ 7), "
        "do Netflix Originals produce healthier subscribers?"
    )
 
    high_rated = df[df["imdb_rating"] >= 7].copy()
 
    health = (
        high_rated.groupby("origin_label")
        .agg(
            users               =("user_id",             "nunique"),
            active_rate         =("is_active",           "mean"),
            avg_monthly_spend   =("monthly_spend",       "mean"),
            avg_tenure_days     =("tenure_days",         "mean"),
            avg_watch_decline   =("watch_decline_ratio", "mean"),
            avg_engagement_ratio=("engagement_ratio_7v30","mean"),
            avg_watch_last30    =("watch_last_30d",      "mean"),
        )
        .reset_index()
        .round(3)
    )
 
    _uc2_render_kpi_cards(health)
    st.divider()
    _uc2_plot_health_bar(health)
    _uc2_plot_active_rate_by_bucket(df)
 
 
def _uc2_plot_yield_box(title_df: pd.DataFrame) -> None:
    """Boxplot of sessions per title by origin.
 
    Args:
        title_df (pd.DataFrame): Title-level dataframe.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    data_orig = title_df[title_df["origin_label"] == "Netflix Original"]["total_sessions"]
    data_lic  = title_df[title_df["origin_label"] == "Licensed"]["total_sessions"]
 
    bp = ax.boxplot(
        [data_orig, data_lic],
        labels=["Netflix Original", "Licensed"],
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
    )
    bp["boxes"][0].set_facecolor(UC2_TEAL)
    bp["boxes"][1].set_facecolor(UC2_AMBER)
    for patch in bp["boxes"]:
        patch.set_alpha(0.75)
 
    ax.set_ylabel("Total Sessions per Title")
    ax.set_title("Session Yield per Title by Origin",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
 
 
def _uc2_plot_yield_scatter(title_df: pd.DataFrame) -> None:
    """Scatterplot of IMDb rating vs total sessions by origin.
 
    Args:
        title_df (pd.DataFrame): Title-level dataframe.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for origin, color in UC2_ORIGIN_COLORS.items():
        sub = title_df[title_df["origin_label"] == origin]
        ax.scatter(sub["imdb_rating"], sub["total_sessions"],
                   alpha=0.4, s=20, color=color, label=origin)
    ax.set_xlabel("IMDb Rating")
    ax.set_ylabel("Total Sessions")
    ax.set_title("IMDb Rating vs Session Yield per Title",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)
 
 
def _uc2_plot_yield_lines(title_df: pd.DataFrame) -> None:
    """Line plots for avg sessions and watch minutes by origin and IMDb bucket.
 
    Args:
        title_df (pd.DataFrame): Title-level dataframe.
    """
    yield_agg = (
        title_df.groupby(["origin_label", "imdb_bucket"], observed=True)
        .agg(
            avg_sessions     =("total_sessions",      "mean"),
            avg_watch_minutes=("total_watch_minutes", "mean"),
            avg_viewers      =("unique_viewers",      "mean"),
            title_count      =("movie_id",            "count"),
        )
        .reset_index()
        .round(1)
    )
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, metric, label in zip(
        axes,
        ["avg_sessions", "avg_watch_minutes"],
        ["Avg Sessions per Title", "Avg Total Watch Minutes per Title"],
    ):
        for origin, color in UC2_ORIGIN_COLORS.items():
            sub = yield_agg[yield_agg["origin_label"] == origin]
            ax.plot(sub["imdb_bucket"].astype(str), sub[metric],
                    marker="o", linewidth=2.5, markersize=7,
                    color=color, label=origin)
        ax.set_xlabel("IMDb Rating Bucket")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by Quality Tier",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)
 
 
def _uc2_render_content_yield(title_df: pd.DataFrame) -> None:
    """Render content yield analysis (sessions and watch minutes per title).
 
    Args:
        title_df (pd.DataFrame): Title-level dataframe.
    """
    st.subheader("3️⃣ Content Yield per Title")
    st.caption(
        "Which titles punch above their weight? Sessions and watch minutes per title, "
        "by origin and quality."
    )
 
    col1, col2 = st.columns(2)
    with col1:
        _uc2_plot_yield_box(title_df)
    with col2:
        _uc2_plot_yield_scatter(title_df)
 
    _uc2_plot_yield_lines(title_df)
 
    st.markdown("**Top 15 Titles by Total Sessions**")
    top_titles = (
        title_df.sort_values("total_sessions", ascending=False)
        .head(15)[
            ["title", "origin_label", "imdb_rating", "genre_primary",
             "total_sessions", "unique_viewers", "avg_completion"]
        ]
        .round(2)
        .reset_index(drop=True)
    )
    st.dataframe(top_titles, use_container_width=True)
 
 
def _uc2_render_genre_origin(df: pd.DataFrame, title_df: pd.DataFrame) -> None:
    """Render genre-level analysis comparing Netflix Originals vs Licensed content.
 
    Args:
        df (pd.DataFrame): Session-level dataframe.
        title_df (pd.DataFrame): Title-level dataframe.
    """
    st.subheader("4️⃣ Genre Breakdown by Origin Type")
    st.caption(
        "Where do Netflix Originals outperform licensed content at the genre level?"
    )
 
    genre_agg = (
        df.groupby(["origin_label", "genre_primary"], observed=True)
        .agg(
            avg_completion=("progress_percentage",    "mean"),
            avg_duration  =("watch_duration_minutes", "mean"),
            sessions      =("session_id",             "count"),
        )
        .reset_index()
        .round(2)
    )
 
    pivot_completion = genre_agg.pivot_table(
        index="genre_primary", columns="origin_label",
        values="avg_completion", aggfunc="mean",
    ).round(1)
 
    if "Netflix Original" in pivot_completion.columns and "Licensed" in pivot_completion.columns:
        pivot_completion["gap (Orig − Lic)"] = (
            pivot_completion["Netflix Original"] - pivot_completion["Licensed"]
        ).round(1)
        pivot_completion = pivot_completion.sort_values("gap (Orig − Lic)", ascending=False)
 
    fig, ax = plt.subplots(figsize=(8, max(5, len(pivot_completion) * 0.45)))
    sns.heatmap(
        pivot_completion, ax=ax, annot=True, fmt=".1f",
        cmap="RdYlGn", center=0,
        linewidths=0.4, cbar_kws={"label": "Avg Completion %"},
    )
    ax.set_title(
        "Completion % by Genre × Origin  (gap = Original − Licensed)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Genre")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    st.pyplot(fig)
    plt.close(fig)
 
    title_counts = (
        title_df.groupby(["origin_label", "genre_primary"], observed=True)
        .agg(title_count=("movie_id", "count"))
        .reset_index()
    )
    pivot_count = title_counts.pivot_table(
        index="genre_primary", columns="origin_label",
        values="title_count", aggfunc="sum", fill_value=0,
    )
 
    fig, ax = plt.subplots(figsize=(8, max(5, len(pivot_count) * 0.45)))
    sns.heatmap(pivot_count, ax=ax, annot=True, fmt=".0f", cmap="Blues",
                linewidths=0.4, cbar_kws={"label": "Number of Titles"})
    ax.set_title(
        "Title Count by Genre × Origin  (volume vs quality trade-off)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    st.pyplot(fig)
    plt.close(fig)
 
    with st.expander("📋 Full genre breakdown"):
        st.dataframe(genre_agg, use_container_width=True)
        
def show_content_investment(artifacts):
    st.header("📺 USE CASE 2: Content Investment Optimization")
 
    st.markdown("""
    **Actor:** Sarah, Content Acquisition/Studio Executive
    **Goal:** Track ROI of content library and make renewal decisions
    **Action:** Analyze content quality vs. retention impact
    """)
 
    st.markdown("---")
 
    if not BIGQUERY_AVAILABLE:
        st.info("""
        ### 🚧 Under Development
 
        This dashboard page will include:
 
        **Features to be implemented by team:**
        1. **Quality Elasticity Chart** - Correlation between IMDB ratings and completion rates
        2. **Title Performance Matrix** - Content metadata × user sentiment analysis
        3. **Revenue Exposure Calculator** - LTV impact per content category
        4. **Renewal Recommendations** - Data-driven content acquisition decisions
 
        **Data Sources:**
        - `movies.csv` - Content metadata and ratings
        - `watch_history.csv` - Viewing completion rates
        - `reviews.csv` - User sentiment scores
        - TMDb API - External rating validation
 
        **Filters Available:**
        - Top 20% LTV users
        - Active in last 60 days
        - Rating buckets (8.0+, 7.0-8.0, etc.)
        - Genre categories
        """)
        return
 
    if not PROJECT_ID:
        st.error("""
        ❌ **Missing BigQuery Configuration**
 
        Please create a `.env` file with your PROJECT_ID:
        ```
        PROJECT_ID=your-project-id
        ```
        """)
        return
 
    try:
        df_raw, title_df_raw = load_uc2_data()
        df, title_df         = _uc2_preprocess(df_raw.copy(), title_df_raw.copy())
 
        # ── Inline filters (replaces use_case_2_app sidebar) ──────────────────
        with st.expander("🎛️ Filters", expanded=False):
            col1, col2, col3 = st.columns(3)
 
            with col1:
                genres     = sorted(df["genre_primary"].dropna().unique())
                sel_genres = st.multiselect(
                    "Genre", genres, default=genres, key="uc2_genres"
                )
 
            with col2:
                plans     = sorted(df["subscription_plan"].dropna().unique())
                sel_plans = st.multiselect(
                    "Subscription plan", plans, default=plans, key="uc2_plans"
                )
 
            with col3:
                origin = st.radio(
                    "Content origin",
                    ["All", "Netflix Original", "Licensed"],
                    key="uc2_origin",
                )
 
        df = _uc2_apply_filters(df, sel_genres, sel_plans, origin)
 
        if df.empty:
            st.warning("No data matches current filters.")
            return
 
        _uc2_render_kpis(df, title_df)
        st.divider()
        _uc2_render_quality_origin(df)
        st.divider()
        _uc2_render_subscriber_health(df)
        st.divider()
        _uc2_render_content_yield(title_df)
        st.divider()
        _uc2_render_genre_origin(df, title_df)
 
    except Exception as e:
        st.error(f"""
        ❌ **Error loading content analytics**
 
        Error: {str(e)}
 
        This may be due to:
        - Missing BigQuery credentials
        - Network connectivity issues
        - Missing data in BigQuery tables
 
        Please check your `.env` file and BigQuery setup.
        """)

# ============================================================================
# USE CASE 3: FEATURE ENGAGEMENT (PUJA) 
# ============================================================================

def show_feature_engagement(artifacts):
    st.header("💡 USE CASE 3: Product Feature & User Experience Analysis")
    
    st.markdown("""
    **Actor:** Puja, Product Manager  
    **Goal:** Identify feature pain points and UX issues driving churn  
    **Action:** Analyze search failures → Understand bounce patterns → Prioritize product improvements
    """)
    
    st.markdown("---")
    
    # Check if BigQuery is available
    if not BIGQUERY_AVAILABLE:
        st.error("""
        ❌ **BigQuery Integration Not Available**
        
        This feature requires:
        - `google-cloud-bigquery` package
        - `python-dotenv` package
        
        Install with: `pip install google-cloud-bigquery python-dotenv`
        """)
        return
    
    if not PROJECT_ID:
        st.error("""
        ❌ **Missing BigQuery Configuration**
        
        Please create a `.env` file with your PROJECT_ID:
        ```
        PROJECT_ID=your-project-id
        ```
        """)
        return
    
    try:
        # Get summary metrics
        overall_churn, avg_bounce, avg_search = get_summary_metrics()
        
        # Display Key Metrics
        st.subheader("📊 Key Product Health Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Overall Churn Rate",
                f"{overall_churn:.1f}%",
                delta=None,
                help="Percentage of users at risk of churning"
            )
        
        with col2:
            st.metric(
                "Avg Bounce Rate",
                f"{avg_bounce:.1f}%",
                delta=None,
                help="Users who watch < 5 mins and leave"
            )
        
        with col3:
            st.metric(
                "Search Failure Rate",
                f"{avg_search:.1f}%",
                delta=None,
                help="Searches that return results but get no clicks"
            )
        
        st.markdown("---")
        
        # SECTION 1: Search Analysis
        st.subheader("🔍 INSIGHT 1: Search Experience Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Search Failure Rate by User Group")
            df_search = load_null_search_analysis()
            
            if not df_search.empty:
                fig = go.Figure()
                
                colors = ['#E50914' if label == 'Churn' else '#00A86B' 
                         for label in df_search['churn_label']]
                
                fig.add_trace(go.Bar(
                    x=df_search['churn_label'],
                    y=df_search['avg_search_failure_rate'],
                    marker_color=colors,
                    text=df_search['avg_search_failure_rate'].round(1),
                    texttemplate='%{text}%',
                    textposition='outside'
                ))
                
                fig.update_layout(
                    xaxis_title="User Group",
                    yaxis_title="Search Failure Rate (%)",
                    template="plotly_dark",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Key insight
                churn_rate = df_search[df_search['churn_label'] == 'Churn']['avg_search_failure_rate'].values[0]
                no_churn_rate = df_search[df_search['churn_label'] == 'Not Churn']['avg_search_failure_rate'].values[0]
                difference = churn_rate - no_churn_rate
                
                st.info(f"""
                **💡 Insight:** Churning users have **{difference:.1f}% higher** search failure rate.  
                This suggests search quality issues contribute to churn.
                """)
            else:
                st.warning("No search data available")
        
        with col2:
            st.markdown("### Top 10 Failed Search Queries (Churn Group)")
            df_failed = load_failed_queries()
            
            if not df_failed.empty:
                # Display as styled dataframe
                st.dataframe(
                    df_failed,
                    use_container_width=True,
                    height=400
                )
                
                st.warning(f"""
                **⚠️ Action Required:** Top failed query: **"{df_failed.iloc[0]['search_query']}"** 
                with **{df_failed.iloc[0]['failure_count']:,}** failed searches.  
                Recommend improving search results or content availability.
                """)
            else:
                st.warning("No failed query data available")
        
        st.markdown("---")
        
        # SECTION 2: Tenure Analysis
        st.subheader("📅 INSIGHT 2: When Do Users Churn?")
        
        df_tenure = load_tenure_analysis()
        
        if not df_tenure.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Use matplotlib plot from usecase3
                fig_tenure = plot_tenure(df_tenure)
                st.pyplot(fig_tenure)
            
            with col2:
                st.markdown("### Key Findings")
                
                # Find the month with highest churn
                max_churn_month = df_tenure.loc[df_tenure['churn_rate'].idxmax()]
                
                st.markdown(f"""
                **Highest Churn Period:**
                - **Month {int(max_churn_month['tenure_months'])}**
                - Churn Rate: **{max_churn_month['churn_rate']:.1f}%**
                - Affected Users: **{int(max_churn_month['total_users']):,}**
                
                **Pattern:**
                - Early months show higher volatility
                - Long-term subscribers (>12 months) stabilize
                
                **Recommendation:**
                - Focus onboarding improvements
                - Introduce engagement hooks at month 1-3
                - Monitor first 90 days closely
                """)
        else:
            st.warning("No tenure data available")
        
        st.markdown("---")
        
        # SECTION 3: Content Bounce Analysis
        st.subheader("🎬 INSIGHT 3: Content Engagement by Genre")
        
        df_bounce = load_final_pm_report()
        
        if not df_bounce.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Use matplotlib plot from usecase3
                fig_bounce = plot_pm_report(df_bounce)
                st.pyplot(fig_bounce)
            
            with col2:
                st.markdown("### Bounce Rate Analysis")
                
                # Find genre with highest bounce rate for churners
                churn_bounce = df_bounce[df_bounce['churn_label'] == 'Churn']
                worst_genre = churn_bounce.loc[churn_bounce['bounce_rate'].idxmax()]
                
                st.markdown(f"""
                **Highest Bounce Genre (Churn Users):**
                - **{worst_genre['genre_primary']}**
                - Bounce Rate: **{worst_genre['bounce_rate']:.1%}**
                
                **What is Bounce Rate?**
                - Users who watch < 5 minutes
                - Indicates content dissatisfaction
                
                **Action Items:**
                - Review content quality in this genre
                - Improve thumbnails/descriptions
                - Adjust recommendation algorithm
                """)
        else:
            st.warning("No bounce rate data available")
        
        st.markdown("---")
        
        # SECTION 4: Feature Importance
        st.subheader("🎯 INSIGHT 4: Top Drivers of Churn")
        
        try:
            fig_importance = plot_feature_importance_from_csv()
            st.pyplot(fig_importance)
            
            st.success("""
            **💡 Strategic Takeaway:**  
            Focus product improvements on the top 3 factors:
            1. **Engagement Drop** - Build re-engagement campaigns
            2. **Days Since Last Watch** - Trigger notifications after 5+ days
            3. **30-Day Viewing Time** - Increase content recommendations
            """)
        except FileNotFoundError:
            st.warning("⚠️ Feature importance file not found. Please ensure model has been trained.")
        
        st.markdown("---")
        
        # Export capability
        st.subheader("📥 Export Analysis Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Search Analysis", use_container_width=True):
                df_search_export = load_null_search_analysis()
                if not df_search_export.empty:
                    csv = df_search_export.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "search_failure_analysis.csv",
                        "text/csv"
                    )
        
        with col2:
            if st.button("📈 Export Tenure Analysis", use_container_width=True):
                df_tenure_export = load_tenure_analysis()
                if not df_tenure_export.empty:
                    csv = df_tenure_export.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "tenure_churn_analysis.csv",
                        "text/csv"
                    )
        
        with col3:
            if st.button("🎬 Export Bounce Analysis", use_container_width=True):
                df_bounce_export = load_final_pm_report()
                if not df_bounce_export.empty:
                    csv = df_bounce_export.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "genre_bounce_analysis.csv",
                        "text/csv"
                    )
    
    except Exception as e:
        st.error(f"""
        ❌ **Error loading product analytics**
        
        Error: {str(e)}
        
        This may be due to:
        - Missing BigQuery credentials
        - Network connectivity issues
        - Missing data in BigQuery tables
        
        Please check your `.env` file and BigQuery setup.
        """)


# ============================================================================
# MODEL PERFORMANCE PAGE
# ============================================================================

def show_model_performance(artifacts):
    st.header("📊 Model Performance Metrics")
    
    metrics = artifacts.get('metrics', {})
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'accuracy' in metrics:
            st.metric(
                "Accuracy",
                f"{metrics['accuracy']:.1%}",
                help="Overall prediction accuracy"
            )
        else:
            st.metric("Accuracy", "N/A")
    
    with col2:
        if 'recall' in metrics:
            st.metric(
                "Recall",
                f"{metrics['recall']:.1%}",
                help="% of actual churners correctly identified"
            )
        else:
            st.metric("Recall", "N/A")
    
    with col3:
        if 'f1_score' in metrics:
            st.metric(
                "F1 Score",
                f"{metrics['f1_score']:.1%}",
                help="Harmonic mean of precision and recall"
            )
        else:
            st.metric("F1 Score", "N/A")
    
    with col4:
        if 'classification_report' in metrics and '0' in metrics['classification_report']:
            precision = metrics['classification_report']['0']['precision']
            st.metric(
                "Precision",
                f"{precision:.1%}",
                help="% of predicted churners that are actually churners"
            )
        else:
            st.metric("Precision", "N/A")
    
    st.markdown("---")
    
    # Classification Report
    st.subheader("Classification Report")
    
    if 'classification_report' in metrics:
        report = metrics['classification_report']
        
        if '0' in report and '1' in report:
            report_df = pd.DataFrame({
                'Class': ['No Churn (0)', 'Churn Risk (1)'],
                'Precision': [report['0']['precision'], report['1']['precision']],
                'Recall': [report['0']['recall'], report['1']['recall']],
                'F1-Score': [report['0']['f1-score'], report['1']['f1-score']],
                'Support': [int(report['0']['support']), int(report['1']['support'])]
            })
            
            st.dataframe(
                report_df,
                use_container_width=True,
                hide_index=True
            )
            
            if 'threshold' in metrics:
                st.info(f"""
                **Model Threshold:** {metrics['threshold']}  
                Optimized for **high recall** to minimize false negatives (missed churners).
                """)
        else:
            st.warning("⚠️ Classification report format not recognized")
    else:
        st.warning("⚠️ Classification report not available in model artifacts")
    
    # Feature Importance
    st.markdown("---")
    st.subheader("Top 15 Most Important Features")
    
    if 'feature_importance' in artifacts and not artifacts['feature_importance'].empty:
        top_features = artifacts['feature_importance'].head(15)
        
        fig = go.Figure(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker_color='#E50914',
            text=top_features['importance'].apply(lambda x: f'{x:.4f}'),
            textposition='outside'
        ))
        
        fig.update_layout(
            xaxis_title="Feature Importance Score",
            yaxis_title="",
            height=500,
            template="plotly_dark",
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Feature importance data not available")


# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()
