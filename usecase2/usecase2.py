# usecase2/usecase2.py
import pandas as pd
import numpy as np
from google.cloud import bigquery
import streamlit as st

# 공통 설정값
TEAL, BLUE, AMBER, RED, GRAY = "#1D9E75", "#185FA5", "#BA7517", "#A32D2D", "#5F5E5A"
ORIGIN_COLORS = {"Netflix Original": TEAL, "Licensed": AMBER}
PROJECT = "netflix-user-behavior"

@st.cache_data(show_spinner="Loading data from BigQuery...")
def load_data():
    """Load session-level and title-level data from BigQuery.
    Returns:
        df      : session-level joined table (watch + movies + churn)
        title_df: title-level aggregated yield table
    """
    client = bigquery.Client(project=PROJECT)
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

    df = client.query(session_sql).result().to_dataframe()
    title_df = client.query(title_sql).result().to_dataframe()
    return df, title_df

def preprocess(df, title_df):
    """Add derived columns like IMDb bucket and origin label.

    Args:
        df (pd.DataFrame): Session-level dataframe
        title_df (pd.DataFrame): Title-level dataframe

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Updated dataframes
    """
    imdb_bins = [0, 5, 6, 7, 8, 9, 10]
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