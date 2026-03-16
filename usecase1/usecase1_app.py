import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import usecase1 as uc1

def run_usecase_1(artifacts):
    st.header("🎯 Targeted Marketing Campaigns")
    
    # 1. 데이터 로드 (Real BigQuery Data)
    with st.spinner("Fetching real-time churn data..."):
        raw_data = uc1.load_high_risk_data()

    # 2. STEP 1: 필터 설정 (UI)
    st.subheader("STEP 1: Set Risk Criteria")
    c1, c2, c3 = st.columns(3)
    with c1:
        threshold = st.slider("Churn Threshold (%)", 50, 100, 85)
    with c2:
        segments = st.multiselect("Segments", options=raw_data['segment'].unique(), default=raw_data['segment'].unique())
    with c3:
        revenue = st.number_input("Min Revenue ($)", 0.0, 100.0, 0.0)

    # 3. 데이터 가공
    filtered_df = uc1.filter_users(raw_data, threshold, segments, revenue)
    metrics = uc1.calculate_impact_metrics(filtered_df)

    # 4. STEP 2: Metrics 표시
    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🚨 High-Risk Users", f"{metrics['count']:,}")
    m2.metric("💵 MRR at Risk", f"${metrics['monthly_revenue_at_risk']:,.2f}")
    m3.metric("📉 Avg Churn Prob", f"{metrics['avg_churn_prob']:.1%}")
    m4.metric("💎 LTV at Risk", f"${metrics['ltv_at_risk']:,.0f}")
    m5.metric("⚠️ Expected Loss", f"${metrics['expected_loss']:,.2f}")

    # 5. STEP 3: 시각화 (Plotly)
    st.divider()
    st.subheader("STEP 3: Segment Breakdown")
    # (여기에는 기존의 Plotly 코드들을 filtered_df를 사용해 배치)
    # 예: render_plotly_charts(filtered_df) 호출

    # 6. STEP 4: Export (CSV)
    st.divider()
    st.subheader("STEP 4: Export Segment")
    col_preview, col_export = st.columns([2, 1])
    
    with col_preview:
        st.dataframe(filtered_df.head(10))
        
    with col_export:
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name="campaign_list.csv", mime="text/csv")