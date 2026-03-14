"""
Use Case 1: High-Risk Churner Dashboard
Run:  python -m streamlit run churn_dashboard.py
Auth: gcloud auth application-default login
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import storage
import pickle
import shap

PROJECT_ID = "netflix-user-behavior"
bucket_name = "netflix-churn-models"
artifact_prefix = "model_outputs/"
RFE_COLUMNS = [
    "days_since_last_watch",
    "total_sessions",
    "avg_completion_rate"
]

st.set_page_config(page_title="Churn Risk Dashboard", page_icon="📉", layout="wide")

@st.cache_resource(show_spinner="Loading model artifacts from GCS...")
def load_artifacts():
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket("netflix-churn-models")

    def load_pickle(file_name):
        blob = bucket.blob(f"model_outputs/{file_name}")
        return pickle.loads(blob.download_as_bytes())

    rf_model = load_pickle("rf_model.pkl")
    scaler = load_pickle("scaler.pkl")
    rfe_scaler = load_pickle("rfe_scaler.pkl")
    kmeans_model = load_pickle("kmeans.pkl")
    feature_names = load_pickle("feature_names.pkl")
    metrics = load_pickle("metrics.pkl")

    segment_profile = pd.read_csv(
        "gs://netflix-churn-models/model_outputs/segment_profile.csv"
    )

    feature_importance = pd.read_csv(
        "gs://netflix-churn-models/model_outputs/feature_importance.csv"
    )

    return (
        rf_model,
        scaler,
        rfe_scaler,
        kmeans_model,
        feature_names,
        metrics,
        segment_profile,
        feature_importance
    )
# @st.cache_data(show_spinner="Training churn model...")
# def get_model(_df):
    #return train_model(_df)

#@st.cache_data(show_spinner="Scoring users...")
#def get_scores(_df, _result):
 #   return score_users(_df, _result)

#@st.cache_data(show_spinner="Computing SHAP values...")
#def get_shap(_result):
 #   return get_shap_values(_result)
 


rf_model, scaler, rfe_scaler, kmeans_model, feature_names, metrics, segment_profile, feature_importance = load_artifacts()
from google.cloud import bigquery

@st.cache_data(show_spinner="Loading feature table...")
def load_features():
    client = bigquery.Client(project=PROJECT_ID)

    query = """
    SELECT *
    FROM `netflix-user-behavior.kaggle_cleaned.churn_features`
    """

    df = client.query(query).to_dataframe()
    df.fillna(0, inplace=True)

    return df

df = load_features()
# recreate segmentation features
x_rfe = df[RFE_COLUMNS]

# scale with saved scaler
x_scaled = rfe_scaler.transform(x_rfe)

# predict segment using saved kmeans model
df["segment"] = kmeans_model.predict(x_scaled)
X = df[feature_names]

missing = set(feature_names) - set(df.columns)

for col in missing:
    df[col] = 0

X = df[feature_names]
X_scaled = scaler.transform(X)
df["churn_probability"] = rf_model.predict_proba(X_scaled)[:, 1]



explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(X_scaled)[1]

feature_names = X.columns


# Header
st.title("Churn Risk Dashboard")
st.caption(f" {len(df):,} users loaded from BigQuery")
st.divider()


# Controls
col1, col2 = st.columns([2, 1])
with col1:
    threshold = st.slider("Churn Probability Threshold", 50, 99, 85, 1, format="%d%%")
with col2:
    plan_filter = st.multiselect(
        "Filter by Plan",
        options=sorted(df["subscription_plan"].unique()),
        default=list(df["subscription_plan"].unique()),
    )

filtered = df[
    (df["churn_probability"] >= threshold / 100) &
    (df["subscription_plan"].isin(plan_filter))
].copy().sort_values("churn_probability", ascending=False)


# KPI cards 
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("High-Risk Users", f"{len(filtered):,}")
with k2:
    st.metric("Revenue at Risk", f"${filtered['monthly_spend'].sum():,.2f}/mo")
with k3:
    avg = filtered["churn_probability"].mean() if len(filtered) else 0
    st.metric("Avg Churn Prob", f"{avg*100:.1f}%")
with k4:
    st.metric("% of User Base", f"{len(filtered)/len(df)*100:.1f}%")

st.divider()


#  Feature Importance + Risk Distribution 
st.subheader("Model Explainability")
fi_col, dist_col = st.columns(2)

with fi_col:
    st.markdown("**Feature Importance (mean |SHAP|)**")
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": mean_shap})
        .sort_values("importance", ascending=True)
        .tail(10)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(importance_df["feature"], importance_df["importance"], color="#e05c5c")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 10 Features Driving Churn")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with dist_col:
    st.markdown("**Risk Distribution (all users)**")
    bins = pd.cut(
        df["churn_probability"],
        bins=[0, 0.5, 0.75, 0.9, 1.0],
        labels=["Low (<50%)", "Medium (50–75%)", "High (75–90%)", "Critical (>90%)"]
    )
    dist = bins.value_counts().reindex(["Low (<50%)", "Medium (50–75%)", "High (75–90%)", "Critical (>90%)"])
    st.bar_chart(dist)

st.divider()


# User table 
st.subheader(f"High-Risk Users ({len(filtered):,})")

display_cols = ["user_id", "first_name", "last_name", "email",
                "subscription_plan", "monthly_spend", "churn_probability",
                "days_since_last_watch", "watch_mean", "completion_rate_mean"]
display_cols = [c for c in display_cols if c in filtered.columns]

show_df = filtered[display_cols].copy()
show_df["churn_probability"]    = (show_df["churn_probability"] * 100).round(1).astype(str) + "%"
show_df["monthly_spend"]        = show_df["monthly_spend"].map("${:.2f}".format)
show_df["watch_mean"]           = show_df["watch_mean"].round(1)
show_df["completion_rate_mean"] = (show_df["completion_rate_mean"] * 100).round(1).astype(str) + "%"

st.dataframe(show_df, use_container_width=True, height=350)


# Per-user SHAP explanation 
st.divider()
st.subheader("🔎 Why is this user at risk?")
st.caption("Select a user to see which factors are driving their churn probability.")

user_ids = filtered["user_id"].tolist()
if user_ids:
    selected_id = st.selectbox("Select User", user_ids)
    user_row    = df[df["user_id"] == selected_id]

    if not user_row.empty:
        user_features = user_row[feature_names]
        user_scaled = scaler.transform(user_features)
        shap_user = explainer.shap_values(user_scaled)[1][0]
        base_val = explainer.expected_value[1]
        feat_names = feature_names
        feat_vals = user_features.iloc[0].values
        row = user_row.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Churn Probability", f"{row['churn_probability']*100:.1f}%")
        with m2:
            st.metric("Days Since Last Watch", f"{int(row['days_since_last_watch'])}")
        with m3:
            st.metric("Avg Watch Duration", f"{row['watch_mean']:.0f} min")
        with m4:
            st.metric("Plan", row["subscription_plan"])

        shap_df = (
            pd.DataFrame({"feature": feat_names, "shap_value": shap_user, "feature_value": feat_vals})
            .reindex(pd.Series(shap_user).abs().sort_values(ascending=False).index)
            .head(10)
            .sort_values("shap_value")
        )

        colors = ["#e05c5c" if v > 0 else "#5c9ee0" for v in shap_df["shap_value"]]
        labels = [f"{r['feature']}={r['feature_value']:.2f}" for _, r in shap_df.iterrows()]

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.barh(labels, shap_df["shap_value"], color=colors)
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("SHAP value  (red = increases churn risk, blue = decreases)")
        ax2.set_title(f"Churn Explanation — {selected_id}")
        ax2.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.caption(f"Base value: {base_val:.3f}  ·  Final prediction: {row['churn_probability']:.3f}")
else:
    st.info("No users match the current threshold and plan filter.")


#  Export
st.divider()
export_cols = ["user_id", "email", "first_name", "last_name",
               "subscription_plan", "monthly_spend", "churn_probability"]
export_cols = [c for c in export_cols if c in filtered.columns]
export_df   = filtered[export_cols].copy()
export_df["churn_probability"] = export_df["churn_probability"].round(4)
csv = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label=f"⬇️  Export {len(filtered):,} users to CSV",
    data=csv,
    file_name=f"high_risk_churners_{threshold}pct.csv",
    mime="text/csv",
    type="primary",
)
