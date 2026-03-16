import os
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Configuration and Constants
PROJECT_ID = 'netflix-user-behavior'
MODEL_DIR = './model_outputs'
SUBSCRIPTION_PLANS = {'Basic': 9.99, 'Standard': 15.49, 'Premium': 19.99}

def load_artifacts():
    """Load model artifacts and segment profiles from pickle/csv files."""
    artifacts = {}
    artifact_names = ['rf_model', 'scaler', 'rfe_scaler', 'kmeans', 'feature_names', 'config']
    
    for name in artifact_names:
        file_path = os.path.join(MODEL_DIR, f'{name}.pkl')
        with open(file_path, 'rb') as file:
            key = name.replace('_names', '_cols')
            artifacts[key] = pickle.load(file)

    profile_path = os.path.join(MODEL_DIR, 'segment_profile.csv')
    artifacts['segment_profile'] = pd.read_csv(profile_path, index_col=0)
    return artifacts

def load_users_from_bq():
    """Retrieve user data from Google BigQuery."""
    from google.cloud import bigquery
    client = bigquery.Client(project=PROJECT_ID)
    query = "SELECT * FROM `netflix-user-behavior.kaggle_cleaned.churn_features`"
    df = client.query(query).result().to_dataframe()
    return df.fillna(0)

def predict_churn_logic(dataframe, artifacts):
    """Logic for predicting churn probability and assigning customer segments."""
    df = dataframe.copy()
    
    # Create segments using RFE features
    rfe_features = ['days_since_last_watch', 'total_sessions', 'avg_completion_rate']
    x_rfe_scaled = artifacts['rfe_scaler'].transform(df[rfe_features])
    df['segment'] = artifacts['kmeans'].predict(x_rfe_scaled)

    # Prepare features: drop non-numeric and ID columns
    drop_cols = ['user_id', 'watch_decline_ratio', 'watch_last_7d', 'total_watch_minutes', 'is_active', 'churn_risk', 'subscription_plan']
    features = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    features = features.select_dtypes(include=[np.number])

    # Align features with training columns
    for col in artifacts['feature_cols']:
        if col not in features.columns: 
            features[col] = 0

    features = features[artifacts['feature_cols']].fillna(0).replace([np.inf, -np.inf], 0).astype(float)
    
    # Generate predictions
    features_scaled = artifacts['scaler'].transform(features)
    df['churn_probability_pct'] = (artifacts['rf_model'].predict_proba(features_scaled)[:, 0] * 100).round(1)
    return df

def get_churn_dist_plot(filtered_data):
    """Generate Churn Probability Distribution Histogram."""
    fig = go.Figure(go.Histogram(
        x=filtered_data['churn_probability_pct'],
        nbinsx=20, marker_color='#E50914'
    ))
    fig.update_layout(
        title="Churn Probability Distribution", 
        xaxis_title="Churn Probability (%)",
        yaxis_title="Count",
        template="plotly_dark", 
        height=400
    )
    return fig

def get_segment_pie_plot(filtered_data):
    """Generate Customer Segment Distribution Pie Chart."""
    segment_counts = filtered_data['segment'].value_counts()
    segment_labels = [f"Segment {i}" for i in segment_counts.index]
    fig = go.Figure(go.Pie(
        labels=segment_labels, values=segment_counts.values,
        hole=0.4, marker_colors=['#E50914', '#B20710', '#8B0000', '#FF1744']
    ))
    fig.update_layout(title="Distribution by Segment", template="plotly_dark", height=400)
    return fig