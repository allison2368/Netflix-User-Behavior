import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

st.title("Customer & Content Analytics Dashboard")

# -----------------------
# Load Data (sample 100 rows)
# -----------------------
@st.cache_data
def load_data():
    users = pd.read_csv("data/cleaned/users.csv").head(100)
    movies = pd.read_csv("data/cleaned/movies.csv").head(100)
    return users, movies

users_df, movies_df = load_data()

# -----------------------
# Train Simple Churn Model
# -----------------------
# We use is_active (1 = active, 0 = churned)
# So churn = 1 - is_active

users_df["churn"] = 1 - users_df["is_active"]

# Select numeric columns for quick demo
numeric_cols = users_df.select_dtypes(include="number").columns.tolist()

# Remove target column
feature_cols = [col for col in numeric_cols if col not in ["is_active", "churn"]]

X = users_df[feature_cols]
y = users_df["churn"]

# Remove NA values (just for the demo)
model_df = pd.concat([X, y], axis=1).dropna()

X_clean = model_df[feature_cols]
y_clean = model_df["churn"]

# Train model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_clean, y_clean)

users_df["churn_probability"] = None
users_df.loc[X_clean.index, "churn_probability"] = model.predict_proba(X_clean)[:, 1]
# -----------------------
# Sidebar
# -----------------------
analysis_type = st.sidebar.radio(
    "Select Analysis",
    ["Churn Analysis", "Content Analysis"]
)

# -----------------------
# CHURN ANALYSIS
# -----------------------
if analysis_type == "Churn Analysis":

    st.header("Churn Analysis")

    churn_threshold = st.sidebar.slider(
        "Select churn probability threshold (%)",
        0, 100, 85
    ) / 100

    filtered_users = users_df[
        users_df["churn_probability"] >= churn_threshold
    ]

    st.metric("High Risk Users", len(filtered_users))

    st.dataframe(filtered_users)

    st.download_button(
        label="Download High Risk Users",
        data=filtered_users.to_csv(index=False),
        file_name="high_risk_users.csv",
        mime="text/csv"
    )

# -----------------------
# CONTENT ANALYSIS
# -----------------------
else:

    st.header("Content Analysis")

    rating_threshold = st.sidebar.slider(
        "Minimum IMDb Rating",
        0.0, 10.0, 8.0
    )

    filtered_movies = movies_df[
        movies_df["imdb_rating"] >= rating_threshold
    ]

    st.metric("Movies Above Rating Threshold", len(filtered_movies))

    st.dataframe(filtered_movies)

    st.download_button(
        label="Download Filtered Movies",
        data=filtered_movies.to_csv(index=False),
        file_name="filtered_movies.csv",
        mime="text/csv"
    )