"""
Streamlit application for the Netflix User Behavior dashboard.
"""
import streamlit as st
import usecase3 as uc3


def run_usecase_3():
    """
    Executes the Use Case 3: Churn Driver Diagnosis dashboard logic.
    """
    st.header("🔍 Why users leave?")

    # Initialize Session State for click-based navigation
    if "detail_view" not in st.session_state:
        st.session_state.detail_view = "Main"

    # Fetch real data for the buttons
    with st.spinner("Fetching latest metrics..."):
        churn_val, bounce_val, search_val = uc3.get_summary_metrics()

    # Overview Metrics (Buttons): These act as clickable cards to trigger different deep-dives.
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(f"🚩 Overall Churn\n\n{churn_val:.1f}%"):
            st.session_state.detail_view = "Tenure"

    with col2:
        if st.button(f"📉 Avg. Bounce Rate\n\n{bounce_val:.1f}%"):
            st.session_state.detail_view = "Bounce"

    with col3:
        if st.button(f"🔎 Search Failure\n\n{search_val:.1f}%"):
            st.session_state.detail_view = "Search"

    st.divider()

    # Dynamic Content Rendering
    if st.session_state.detail_view == "Main":
        st.info("👆 Click the cards above to explore detailed behavioral analysis.")
        # Default view: Displaying the most significant driver (Bounce Rate)
        df_bounce = uc3.load_final_pm_report()

    elif st.session_state.detail_view == "Tenure":
        st.subheader("⏳ Subscription Tenure Analysis")
        df_tenure = uc3.load_tenure_analysis()
        st.pyplot(uc3.plot_tenure(df_tenure))
        if st.button("Back to Overview"):  # button that going back to overview
            st.session_state.detail_view = "Main"
            st.rerun()

    elif st.session_state.detail_view == "Bounce":
        st.subheader("🎯 Deep-Dive: Genre-specific Bounce Rate")
        df_bounce = uc3.load_final_pm_report()
        st.pyplot(uc3.plot_pm_report(df_bounce))
        if st.button("Back to Overview"):
            st.session_state.detail_view = "Main"
            st.rerun()

    elif st.session_state.detail_view == "Search":
        st.subheader("🔎 Search Fatigue Analysis")
        df_null = uc3.load_null_search_analysis()
        st.pyplot(uc3.plot_null_search(df_null))

        st.write("😡 **Top 10 Failed Queries for Churn Group:**")
        df_queries = uc3.load_failed_queries()
        st.table(df_queries)
        if st.button("Back to Overview"):  # button that going back to overview
            st.session_state.detail_view = "Main"
            st.rerun()


# Execution
if __name__ == "__main__":
    run_usecase_3()
