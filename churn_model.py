"""
churn_model.py — shared production code
Used by both churn_dashboard.py and the notebook.

Contains:
  - load_data()         : pulls from BigQuery
  - engineer_features() : all 7 features from the notebook
  - train_model()       : SMOTE + XGBoost, returns result dict
  - score_users()       : scores all users using a trained model
"""

import numpy as np
import pandas as pd
from google.cloud import bigquery

# Big Query config
PROJECT_ID = "netflix-user-behavior"
DATASET    = "kaggle_cleaned"


# Load data from BigQuery
def load_data():
    """Pull watch history + movies + users from BigQuery."""
    client = bigquery.Client(project=PROJECT_ID)

    sql_watch = f"""
        SELECT m.movie_id, m.duration_minutes,
               w.user_id, w.watch_date, w.watch_duration_minutes
        FROM `{PROJECT_ID}.{DATASET}.movies_cleaned`  AS m
        JOIN `{PROJECT_ID}.{DATASET}.watch_history_cleaned` AS w
          ON m.movie_id = w.movie_id
    """
    sql_users = f"SELECT * FROM `{PROJECT_ID}.{DATASET}.users_cleaned`"

    pdf  = client.query(sql_watch).result().to_dataframe()
    user = client.query(sql_users).result().to_dataframe()

    return pdf, user


# Feature engineering
def engineer_features(pdf, user):
    """
    Builds all 7 features from the notebook.

    Feature 1: completion_rate       — how much of each movie was watched
    Feature 2: days_since_signup     — tenure at time of each watch event
    Feature 3: days_since_last_watch — recency (high = churn risk)
    Feature 4: recent_7d_watch_sum   — total watch minutes in last 7 days
    Feature 5: watch_decline_ratio   — recent activity vs normal weekly pace
    Feature 6: total_content_value   — total watch time / monthly spend
    Feature 7: total_tenure_days     — full days since signup
    """
    pdf = pdf.copy()
    pdf["watch_date"] = pd.to_datetime(pdf["watch_date"])
    pdf = pdf.sort_values(["user_id", "watch_date"])

    # Feature 1: completion rate per session
    pdf["completion_rate"] = (
        pdf["watch_duration_minutes"] / (pdf["duration_minutes"] + 1e-9)
    ).clip(upper=1.0)

    # Feature 2: days since signup at each watch event
    temp = pdf.merge(
        user[["user_id", "subscription_start_date"]], on="user_id", how="left"
    )
    pdf["days_since_signup"] = (
        pdf["watch_date"] - pd.to_datetime(temp["subscription_start_date"])
    ).dt.days

    latest_date    = pdf["watch_date"].max()
    seven_days_ago = latest_date - pd.Timedelta(days=7)

    # Feature 3: recency
    recency = (
        pdf.groupby("user_id")["watch_date"].max()
        .reset_index()
        .rename(columns={"watch_date": "last_watch_date"})
    )
    recency["days_since_last_watch"] = (
        latest_date - recency["last_watch_date"]
    ).dt.days

    # Feature 4: recent 7-day watch volume
    recent_7d = (
        pdf[pdf["watch_date"] >= seven_days_ago]
        .groupby("user_id")["watch_duration_minutes"].sum()
        .reset_index()
        .rename(columns={"watch_duration_minutes": "recent_7d_watch_sum"})
    )

    # Aggregate per-user stats
    stats = pdf.groupby("user_id").agg(
        movie_id_count       = ("movie_id",               "count"),
        total_watch_sum      = ("watch_duration_minutes", "sum"),
        watch_mean           = ("watch_duration_minutes", "mean"),
        completion_rate_mean = ("completion_rate",        "mean"),
        max_tenure_at_watch  = ("days_since_signup",      "max"),
    ).reset_index()

    features = (
        stats
        .merge(recency[["user_id", "days_since_last_watch"]], on="user_id", how="left")
        .merge(recent_7d, on="user_id", how="left")
        .fillna(0)
    )

    # Feature 5: watch decline ratio
    features["watch_decline_ratio"] = (
        features["recent_7d_watch_sum"] / (features["total_watch_sum"] / 4 + 1e-9)
    )

    # Merge with user profile
    df = user.merge(features, on="user_id", how="left")
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Feature 6: content value
    df["total_content_value"] = df["total_watch_sum"] / (df["monthly_spend"] + 1e-9)

    # Feature 7: total tenure days
    df["subscription_start_date"] = pd.to_datetime(df["subscription_start_date"])
    df["total_tenure_days"] = (
        latest_date - df["subscription_start_date"]
    ).dt.days.clip(lower=0)

    # Churn label: watch_decline_ratio < 0.2 → churn_risk = 1
    df["churn_risk"] = (df["watch_decline_ratio"] < 0.2).astype(int)

    return df


# Exclude these columns
IGNORE_COLS = [
    "user_id", "email", "first_name", "last_name", "household_size",
    "state_province", "city", "country", "location_country",
    "is_active", "monthly_spend", "age", "gender", "created_at",
    "subscription_plan", "primary_device", "imdb_rating_mean",
    "watch_decline_ratio", "recent_7d_watch_sum",
    "churn_risk", "subscription_start_date", "last_watch_date",
]


def _prepare_X(df, feature_cols=None):
    """Drop ignored columns, one-hot encode, align to feature_cols if given."""
    drop_cols = [c for c in IGNORE_COLS if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = pd.get_dummies(X, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    if feature_cols is not None:
        X = X.reindex(columns=feature_cols, fill_value=0)
    return X


# Train model
def train_model(df):
    """
    Trains XGBoost with exact settings from the notebook.

    Returns a dict with everything the notebook's evaluation cells need:
      model              — trained XGBoost model
      scaler             — fitted StandardScaler
      feature_cols       — list of feature column names (for scoring new data)
      X                  — full feature matrix (used by SHAP + LR + RF cells)
      X_train_resampled  — SMOTE-resampled training features
      y_train_resampled  — SMOTE-resampled training labels
      X_test_scaled      — scaled test features
      X_test             — unscaled test features (used by SHAP summary plot)
      y_test             — test labels
    """
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    X = _prepare_X(df)
    y = df["churn_risk"]

    # Train/test split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # StandardScaler — fit on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # SMOTE — exact settings from notebook
    smote = SMOTE(sampling_strategy=0.2, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # XGBoost
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=1,
        random_state=42,
        eval_metric="logloss",
        base_score=0.5,          # fixes SHAP compatibility with XGBoost 2.x
        verbosity=0,
    )
    model.fit(X_train_resampled, y_train_resampled)

    return {
        "model":             model,
        "scaler":            scaler,
        "feature_cols":      X.columns.tolist(),
        "X":                 X,
        "X_train_resampled": X_train_resampled,
        "y_train_resampled": y_train_resampled,
        "X_test_scaled":     X_test_scaled,
        "X_test":            X_test,
        "y_test":            y_test,
    }


# Score users
def score_users(df, result):
    """
    Score every user in df using the trained model + scaler.
    Accepts the dict returned by train_model().
    Returns a Series of churn probabilities (class 0 = churner).
    """
    X_all = _prepare_X(df, feature_cols=result["feature_cols"])
    X_all_scaled = result["scaler"].transform(X_all)
    # class 0 = churner — same convention as notebook (threshold 0.4)
    probs = result["model"].predict_proba(X_all_scaled)[:, 0]
    return pd.Series(probs, index=df.index)

# SHAP values
def get_shap_values(result):
    """SHAP values for the test set — used for the feature importance chart."""
    import shap
    explainer = shap.TreeExplainer(result["model"])
    shap_values = explainer.shap_values(result["X_test_scaled"])
    return shap_values, result["feature_cols"]


def get_shap_for_user(user_row_df, result):
    """SHAP values for a single user — used for the per-user waterfall chart."""
    import shap
    X_user        = _prepare_X(user_row_df, feature_cols=result["feature_cols"])
    X_user_scaled = result["scaler"].transform(X_user)
    explainer = shap.TreeExplainer(result["model"])
    shap_vals     = explainer.shap_values(X_user_scaled)
    base_value    = explainer.expected_value
    return shap_vals[0], base_value, result["feature_cols"], X_user.iloc[0].tolist()