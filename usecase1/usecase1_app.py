import streamlit as st
import pandas as pd
from datetime import datetime
from . import usecase1 as uc1 # Import logic module

def run_usecase_1():
    """Main function for the Use Case 1 Dashboard UI."""
    st.title("🎬 NETFLIX CHURN PREDICTION")
    st.markdown("### Use Case 1: High-Risk Churners Export")
    
    # Load data and artifacts
    artifacts = uc1.load_artifacts()
    raw_data = uc1.load_users_from_bq()
    user_data = uc1.predict_churn_logic(raw_data, artifacts)

    # Set Risk Criteria (Filters)
    st.header("STEP 1: Set Risk Criteria")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        threshold = st.slider("🎯 Churn Threshold (%)", 50, 100, 85)
    with col2:
        num_seg = artifacts['config']['N_CLUSTERS']
        segs = st.multiselect(
            "📊 Target Segments", 
            [f"Segment {i}" for i in range(num_seg)], 
            [f"Segment {i}" for i in range(num_seg)]
        )
    with col3:
        plans = st.multiselect(
            "💳 Subscription Plans", 
            list(uc1.SUBSCRIPTION_PLANS.keys()), 
            list(uc1.SUBSCRIPTION_PLANS.keys())
        )

    # Apply Filtering Logic
    filtered_df = user_data[user_data['churn_probability_pct'] >= threshold].copy()
    seg_ids = [int(s.split()[-1]) for s in segs]
    filtered_df = filtered_df[filtered_df['segment'].isin(seg_ids)]
    
    if 'subscription_plan' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['subscription_plan'].isin(plans)]

    # Metrics Overview
    st.markdown("---")
    st.header("STEP 2: High-Risk Segment Overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("🚨 High-Risk Users", f"{len(filtered_df):,}")
    
    avg_prob = filtered_df['churn_probability_pct'].mean() if len(filtered_df) > 0 else 0
    m2.metric("📊 Avg Probability", f"{avg_prob:.1f}%")
    
    # Calculate revenue at risk (Optional based on plans)
    total_rev = len(filtered_df) * 15.0  # Default avg if plan not found
    m3.metric("💰 Est. Monthly Revenue at Risk", f"${total_rev:,.0f}")
    
    # Visualizations (Calling functions from usecase1.py)
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(uc1.get_churn_dist_plot(filtered_df), use_container_width=True)
    with col_right:
        if len(filtered_df) > 0:
            st.plotly_chart(uc1.get_segment_pie_plot(filtered_df), use_container_width=True)

    # Export Section
    st.markdown("---")
    st.header("STEP 3: Export Target Segment")
    if len(filtered_df) > 0:
        st.subheader("📋 Preview (Top 10)")
        st.dataframe(filtered_df.head(10), use_container_width=True)
        
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download High-Risk User List (CSV)", 
            data=csv, 
            file_name=f"high_risk_churners_{datetime.now().strftime('%Y%m%d')}.csv", 
            mime="text/csv"
        )
    else:
        st.warning("⚠️ No users match the current criteria. Please adjust the filters.")