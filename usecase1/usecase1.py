"""
Netflix Churn Dashboard - Use Case 1: High-Risk Churners Export
Run: streamlit run churn_dashboard_usecase1.py
"""

import os
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

try:
    from google.cloud import bigquery
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

#config

PROJECT_ID = 'netflix-user-behavior'
MODEL_DIR = './model_outputs'
SUBSCRIPTION_PLANS = {'Basic': 9.99, 'Standard': 15.49, 'Premium': 19.99}

#set page config and custom styles

st.set_page_config(page_title="Netflix Churn", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    .main {
        background: linear-gradient(to bottom, #141414 0%, #000000 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #221f1f 0%, #141414 100%);
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    h1, h2, h3 {
        color: #E50914 !important;
    }
    [data-testid="stMetricValue"] {
        color: #E50914;
        font-size: 2rem !important;
    }
    .stButton>button {
        background-color: #E50914;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
    }
    .stButton>button:hover {
        background-color: #B20710;
    }
</style>
""", unsafe_allow_html=True)

# load model artifacts and data with caching to optimize performance

@st.cache_resource
def load_artifacts():
    """Load model artifacts from pickle files."""
    artifacts = {}
    artifact_names = [
        'rf_model', 'scaler', 'rfe_scaler',
        'kmeans', 'feature_names', 'config'
    ]

    for name in artifact_names:
        file_path = os.path.join(MODEL_DIR, f'{name}.pkl')
        try:
            with open(file_path, 'rb') as file:
                key = name.replace('_names', '_cols')
                artifacts[key] = pickle.load(file)
        except FileNotFoundError as error:
            st.error(f" Model file not found: {file_path}")
            st.error(str(error))
            st.stop()

    profile_path = os.path.join(MODEL_DIR, 'segment_profile.csv')
    try:
        artifacts['segment_profile'] = pd.read_csv(
            profile_path, index_col=0
        )
    except FileNotFoundError as error:
        st.error(f"Segment profile not found: {profile_path}")
        st.error(str(error))
        st.stop()

    return artifacts

@st.cache_data
def load_users():
    """Load user data from BigQuery - NO FALLBACK."""
    if not BIGQUERY_AVAILABLE:
        st.error("❌ BigQuery library not installed!")
        st.error("Install with: pip install google-cloud-bigquery")
        st.stop()
    
    try:
        client = bigquery.Client(project=PROJECT_ID)
        query = ("SELECT * FROM "
                "`netflix-user-behavior.kaggle_cleaned.churn_features`")
        dataframe = client.query(query).result().to_dataframe()
        dataframe.fillna(0, inplace=True)
        
        # Success indicator
        st.sidebar.success(f"✅ Loaded {len(dataframe):,} users from BigQuery")
        
        return dataframe
        
    except Exception as error:
        st.error("❌ Failed to load data from BigQuery!")
        st.error(f"Error: {error}")
        st.info("""
        **Possible solutions:**
        1. Authenticate: `gcloud auth application-default login`
        2. Check project ID: `netflix-user-behavior`
        3. Verify table exists in BigQuery console
        4. Check you have read permissions
        """)
        st.stop()

def predict_churn(dataframe, artifacts):
    """Predict churn probability for all users."""
    dataframe = dataframe.copy()

    # Create segments
    rfe_features = [
        'days_since_last_watch',
        'total_sessions',
        'avg_completion_rate'
    ]
    x_rfe = dataframe[rfe_features]
    x_rfe_scaled = artifacts['rfe_scaler'].transform(x_rfe)
    dataframe['segment'] = artifacts['kmeans'].predict(x_rfe_scaled)

    # Prepare features - drop non-numeric and ID columns
    drop_cols = [
        'user_id', 'watch_decline_ratio', 'watch_last_7d',
        'total_watch_minutes', 'is_active', 'churn_risk',
        'subscription_plan'
    ]
    features = dataframe.drop(
        columns=[c for c in drop_cols if c in dataframe.columns],
        errors='ignore'
    )

    # Ensure only numeric columns
    features = features.select_dtypes(include=[np.number])

    # Ensure matching features with training
    for col in artifacts['feature_cols']:
        if col not in features.columns:
            features[col] = 0

    features = features[artifacts['feature_cols']]
    features = features.fillna(0).replace(
        [np.inf, -np.inf], 0
    ).astype(float)

    # Predict
    features_scaled = artifacts['scaler'].transform(features)
    churn_probs = artifacts['rf_model'].predict_proba(
        features_scaled
    )[:, 0]
    dataframe['churn_probability_pct'] = (churn_probs * 100).round(1)

    return dataframe
#dashboard main function and UI components

def main():
    """Main dashboard function."""
    # Header
    st.title("🎬 NETFLIX CHURN PREDICTION")
    st.markdown("### Use Case 1: High-Risk Churners Export")

    st.markdown("---")

    # Sidebar
    st.sidebar.title("📍 Use Case")
    st.sidebar.markdown("""
    **Actor:** Marcus, Marketing Manager

    **Objective:** Identify high churn probability users

    **Action:** Export for discount campaigns
    """)

    # Load data
    artifacts = load_artifacts()
    user_data = load_users()
    user_data = predict_churn(user_data, artifacts)

    # Get filter inputs
    filters = get_filter_inputs(artifacts)

    # Apply filters and calculate metrics
    filtered_data, metrics = apply_filters_and_calculate(
        user_data, filters
    )

    # Display results
    display_step2_results(filtered_data, metrics, user_data)

    # Display visualizations
    display_visualizations(filtered_data)

    # Display export section
    display_step3_export(filtered_data)

    # Display segment profiles
    display_segment_profiles(artifacts)


def get_filter_inputs(artifacts):
    """Get user filter inputs from UI."""
    st.header("STEP 1: Set Risk Criteria")

    col1, col2, col3 = st.columns(3)

    with col1:
        churn_threshold = st.slider(
            "🎯 Churn Probability Threshold (%)",
            min_value=50,
            max_value=100,
            value=85,
            step=5
        )

    with col2:
        num_segments = artifacts['config']['N_CLUSTERS']
        segment_options = [f"Segment {i}" for i in range(num_segments)]
        selected_segments = st.multiselect(
            "📊 Customer Segments",
            options=segment_options,
            default=segment_options
        )

    with col3:
        selected_plans = st.multiselect(
            "💳 Subscription Plans",
            options=list(SUBSCRIPTION_PLANS.keys()),
            default=list(SUBSCRIPTION_PLANS.keys())
        )

    return {
        'churn_threshold': churn_threshold,
        'segments': selected_segments,
        'plans': selected_plans
    }


def apply_filters_and_calculate(user_data, filters):
    """Apply filters to data and calculate metrics."""
    # Apply churn threshold filter
    filtered_data = user_data[
        user_data['churn_probability_pct'] >= filters['churn_threshold']
    ].copy()

    # Apply segment filter
    segment_ids = [int(seg.split()[-1]) for seg in filters['segments']]
    filtered_data = filtered_data[filtered_data['segment'].isin(segment_ids)]

    # Apply plan filter and calculate revenue
    if 'subscription_plan' in filtered_data.columns:
        filtered_data = filtered_data[
            filtered_data['subscription_plan'].isin(filters['plans'])
        ]
        filtered_data['monthly_revenue'] = (
            filtered_data['subscription_plan'].map(SUBSCRIPTION_PLANS)
        )
        total_revenue = filtered_data['monthly_revenue'].sum()
    else:
        avg_plan_price = np.mean(list(SUBSCRIPTION_PLANS.values()))
        total_revenue = len(filtered_data) * avg_plan_price

    # Calculate metrics
    num_high_risk = len(filtered_data)
    avg_churn_prob = (
        filtered_data['churn_probability_pct'].mean()
        if num_high_risk > 0 else 0
    )

    metrics = {
        'num_high_risk': num_high_risk,
        'avg_churn_prob': avg_churn_prob,
        'total_revenue': total_revenue
    }

    return filtered_data, metrics


def display_step2_results(filtered_data, metrics, user_data):
    """Display Step 2: High-Risk Segment Overview."""
    st.markdown("---")
    st.header("STEP 2: High-Risk Segment Overview")

    col1, col2, col3, col4 = st.columns(4)

    pct_of_base = (
        metrics['num_high_risk'] / len(user_data) * 100
        if len(user_data) > 0 else 0
    )

    col1.metric("🚨 High-Risk Users", f"{metrics['num_high_risk']:,}")
    col2.metric("📊 Avg Churn Probability",
               f"{metrics['avg_churn_prob']:.1f}%")
    col3.metric("💰 Revenue at Risk",
               f"${metrics['total_revenue']:,.2f}/mo")
    col4.metric("📈 % of User Base", f"{pct_of_base:.1f}%")

    st.markdown("---")


def display_visualizations(filtered_data):
    """Display churn probability distribution and segment breakdown."""
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(go.Histogram(
            x=filtered_data['churn_probability_pct'],
            nbinsx=20,
            marker_color='#E50914'
        ))
        fig.update_layout(
            title="Churn Probability Distribution",
            xaxis_title="Churn Probability (%)",
            yaxis_title="Users",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if len(filtered_data) > 0:
            segment_counts = filtered_data['segment'].value_counts()
            segment_labels = [f"Segment {i}" for i in segment_counts.index]

            fig = go.Figure(go.Pie(
                labels=segment_labels,
                values=segment_counts.values,
                hole=0.4,
                marker_colors=['#E50914', '#B20710', '#8B0000', '#FF1744']
            ))
            fig.update_layout(
                title="Distribution by Segment",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)


def display_step3_export(filtered_data):
    """Display Step 3: Export Segment."""
    st.markdown("---")
    st.header("STEP 3: Export Segment")

    if len(filtered_data) > 0:
        display_export_section(filtered_data)
    else:
        st.warning(
            "⚠️ No users match the filters. Try adjusting your criteria."
        )


def display_export_section(filtered_data):
    """Display preview table and export functionality."""
    st.subheader("📋 Preview (Top 10)")

    display_cols = [
        'user_id', 'churn_probability_pct', 'segment',
        'days_since_last_watch', 'avg_completion_rate'
    ]

    if 'subscription_plan' in filtered_data.columns:
        display_cols.extend(['subscription_plan', 'monthly_revenue'])

    preview_data = filtered_data[display_cols].head(10).copy()
    preview_data.columns = [
        col.replace('_', ' ').title() for col in preview_data.columns
    ]

    st.dataframe(
        preview_data.style.background_gradient(
            subset=['Churn Probability Pct'],
            cmap='Reds'
        ),
        use_container_width=True,
        hide_index=True
    )

    # Export button
    col1, col2 = st.columns([2, 8])
    with col1:
        export_data = filtered_data[display_cols].copy()
        export_data.columns = [
            col.replace('_', ' ').title() for col in export_data.columns
        ]
        csv_data = export_data.to_csv(index=False)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"high_risk_churners_{timestamp}.csv",
            mime="text/csv"
        )
    with col2:
        st.metric("Total Rows", f"{len(export_data):,}")


def display_segment_profiles(artifacts):
    """Display customer segment profiles table."""
    st.markdown("---")
    st.header("📊 Customer Segment Profiles")

    profile_data = artifacts['segment_profile'].copy()
    profile_data.index = [f"Segment {i}" for i in profile_data.index]
    profile_data.columns = [
        col.replace('_', ' ').title() for col in profile_data.columns
    ]

    st.dataframe(
        profile_data.style.background_gradient(cmap='RdYlGn_r', axis=0),
        use_container_width=True
    )

    st.info("""
    💡 **Segment Interpretation:**
    - Lower 'Days Since Last Watch' = More active users
    - Higher 'Total Sessions' = More engaged users
    - Higher 'Completion Rate' = Better content satisfaction
    """)


if __name__ == "__main__":
    main()