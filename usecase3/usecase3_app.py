"""
Streamlit application for the Netflix User Behavior dashboard.
"""

import streamlit as st

from usecase3 import usecase3 as uc3


def run_usecase_3():
    """
    Executes the Use Case 3: Churn Driver Diagnosis dashboard logic.
    """
    st.set_page_config(layout="wide")  # wide theme
    st.header("🔍 Why users leave?")

    # Initialize session state
    if "detail_view" not in st.session_state:
        st.session_state.detail_view = "Main"

    with st.spinner("Fetching latest metrics..."):
        churn_val, bounce_val, search_val = uc3.get_summary_metrics()

    # Divide main layout
    left_col, right_col = st.columns([1, 5])

    # Left column: buttons for charts
    with left_col:
        st.subheader("Key Stats")
        st.metric("Overall Churn", f"{churn_val:.1f}%")
        st.metric("Bounce Rate", f"{bounce_val:.1f}%")
        st.metric("Search Fail", f"{search_val:.1f}%")

        st.divider()
        st.write("More Insights")

        if st.button("⏳ Tenure", use_container_width=True):
            st.session_state.detail_view = "Tenure"
        if st.button("🎯 Genre", use_container_width=True):
            st.session_state.detail_view = "Genre"
        if st.button("🔎 Search", use_container_width=True):
            st.session_state.detail_view = "Search"

        if st.session_state.detail_view != "Main":
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.detail_view = "Main"
                st.rerun()

    # Right column: dynamic charts
    with right_col:
        if st.session_state.detail_view == "Main":
            st.subheader("Main Causes of User Churn")
            st.pyplot(uc3.plot_feature_importance_from_csv(), use_container_width=True)

        elif st.session_state.detail_view == "Tenure":
            st.subheader("Subscription Tenure Analysis")
            df_tenure = uc3.load_tenure_analysis()
            st.pyplot(uc3.plot_tenure(df_tenure), use_container_width=True)

        elif st.session_state.detail_view == "Genre":
            st.subheader("Where Users Lose Interest")
            df_bounce = uc3.load_final_pm_report()
            _, chart_col, _ = st.columns([0.5, 4, 0.5])

            with chart_col:
                st.pyplot(uc3.plot_pm_report(df_bounce), use_container_width=True)
            with st.expander("💡 Calculation Logic"):
                st.caption("Bounce Rate = (Views < 5 mins / Total Views) * 100")

        elif st.session_state.detail_view == "Search":
            st.subheader("Top 10 Most Searched (No Results Found)")
            st.table(uc3.load_failed_queries().head(10))


# Execution
if __name__ == "__main__":
    run_usecase_3()
