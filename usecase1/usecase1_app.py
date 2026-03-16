"""
Streamlit UI for Use Case 1: High-Risk Churners Export.
Provides filters, metrics, and CSV export functionality.
"""

from datetime import datetime

import streamlit as st
from . import usecase1 as uc1  # import logic module


def run_usecase_1():  # pylint: disable=too-many-locals
    """Main function for the Use Case 1 Dashboard UI."""
    st.title("Churn Prediction")
    st.markdown(
        "**Marcus' question:** Who are the high-risk churners and what "
        "marketing methods can he implement to draw them back?"
    )
    st.divider()

    # Load data and artifacts
    artifacts = uc1.load_artifacts()
    raw_data = uc1.load_users_from_bq()
    user_data = uc1.predict_churn_logic(raw_data, artifacts)

    # Set Risk Criteria (Filters)
    st.header("STEP 1: Set Risk Criteria")
    st.markdown(
        "We used K-means clustering using **Recency**, **Frequency**, and "
        "**Engagement** segmentation to create interpretable segments."
    )
    st.markdown(
        "- **Segment 0 — Browsing users:** Users that log in relatively "
        "frequently but just browse.\n"
        "- **Segment 1 — Regular viewers:** Most active; they have the "
        "highest average total sessions and relatively high completion rate.\n"
        "- **Segment 2 — Dormant users:** Haven't watched in a while, "
        "likely churned.\n"
        "- **Segment 3 — Quality viewers:** Less frequent watchers but when "
        "they do, they spend a longer time watching."
    )
    st.markdown("")  # small spacing before sliders
    col1, col2, col3 = st.columns(3)

    with col1:
        threshold = st.slider("Churn Threshold (%)", 50, 100, 85)
    with col2:
        num_seg = artifacts["config"]["N_CLUSTERS"]
        segs = st.multiselect(
            "Target Segments",
            [f"Segment {i}" for i in range(num_seg)],
            [f"Segment {i}" for i in range(num_seg)],
        )
    with col3:
        plans = st.multiselect(
            "Subscription Plans",
            list(uc1.SUBSCRIPTION_PLANS.keys()),
            list(uc1.SUBSCRIPTION_PLANS.keys()),
        )

    # Apply Filtering Logic
    filtered_df = user_data[user_data["churn_probability_pct"] >= threshold].copy()
    seg_ids = [int(s.split()[-1]) for s in segs]
    filtered_df = filtered_df[filtered_df["segment"].isin(seg_ids)]

    if "subscription_plan" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["subscription_plan"].isin(plans)]

    # Metrics Overview
    st.markdown("---")
    st.header("STEP 2: High-Risk Segment Overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("High-Risk Users", f"{len(filtered_df):,}")

    avg_prob = (
        filtered_df["churn_probability_pct"].mean() if len(filtered_df) > 0 else 0
    )
    m2.metric("Avg Probability", f"{avg_prob:.1f}%")

    # Calculate revenue at risk (Optional based on plans)
    total_rev = len(filtered_df) * 15.0  # Default avg if plan not found
    m3.metric("Est. Monthly Revenue at Risk", f"${total_rev:,.0f}")

    # Visualizations (Calling functions from usecase1.py)
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(uc1.get_churn_dist_plot(filtered_df), use_container_width=True)
    with col_right:
        if len(filtered_df) > 0:
            st.plotly_chart(
                uc1.get_segment_pie_plot(filtered_df), use_container_width=True
            )

    # Export Section
    st.markdown("---")
    st.header("STEP 3: Export Target Segment")
    st.markdown(
        "**Recommended marketing methods for each segment:**\n"
        "- **Segment 0:** Send personalized recommendations, make it easier "
        "to find content, improve recommendations.\n"
        "- **Segment 1:** Offer premium features, recommend new movies.\n"
        "- **Segment 2:** Send promotions (cheaper deals), send targeted "
        "messages.\n"
        "- **Segment 3:** Personalized recommendations, notify of similar "
        "content."
    )
    if len(filtered_df) > 0:
        st.subheader("Preview (Top 10)")
        st.dataframe(filtered_df.head(10), use_container_width=True)

        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download High-Risk User List (CSV)",
            data=csv,
            file_name=f"high_risk_churners_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.warning("No users match the current criteria. Please adjust the filters.")
