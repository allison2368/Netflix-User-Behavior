"""
Data loading and visualization functions for Netflix Data Dashboard (Use Case 3).
This module handles BigQuery data retrieval and Matplotlib/Seaborn plotting.
"""

import os
import matplotlib

matplotlib.use("Agg")  # pylint: disable=wrong-import-position
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import streamlit as st

from dotenv import load_dotenv
from google.cloud import bigquery

# Load environment variables from .env file
load_dotenv()

# GCP look for GOOGLE_APPLICATION_CREDENTIALS
PROJECT_ID = os.getenv("PROJECT_ID")


@st.cache_data
def load_null_search_analysis():
    """
    Analyzes search failure rates (Results returned but no click)
    split by Churn vs. Not Churn groups.
    """
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


# Churn group is newbies or long term subscribers?
@st.cache_data
def load_tenure_analysis():
    """Groups churn rates by user tenure in months."""
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

    avg_bounce = df_bounce["bounce_rate"].mean() * 100
    avg_search = df_search["avg_search_failure_rate"].mean()
    overall_churn = df_tenure["churn_rate"].mean()

    return overall_churn, avg_bounce, avg_search
