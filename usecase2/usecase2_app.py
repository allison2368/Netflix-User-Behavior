"""
usecase2_app.py
Streamlit application for Content Investment Optimization (Use Case 2).

Imports all data loading and visualization functions from usecase2.py.

Run:
    streamlit run usecase2_app.py
"""

import streamlit as st
from . import usecase2 as uc2  # Import logic module

st.set_page_config(
    page_title="Content Investment · Sarah",
    layout="wide",
)


def run_usecase_2():
    """Run the Content Investment Optimization dashboard."""
    st.title("Content Investment Optimization")
    st.markdown(
        "**Sarah's question:** *Does investing in fewer high-rated Netflix Originals "
        "drive more retention than a higher volume of average-rated licensed content?*"
    )
    st.divider()

    df_raw, title_df_raw = uc2.load_data()
    df, title_df = uc2.preprocess(df_raw.copy(), title_df_raw.copy())

    # Sidebar filters
    st.sidebar.header("Filters")

    genres = sorted(df["genre_primary"].dropna().unique())
    sel_genres = st.sidebar.multiselect("Genre", genres, default=genres)

    plans = sorted(df["subscription_plan"].dropna().unique())
    sel_plans = st.sidebar.multiselect("Subscription plan", plans, default=plans)

    origin = st.sidebar.radio("Content origin", ["All", "Netflix Original", "Licensed"])

    mask = df["genre_primary"].isin(sel_genres) & df["subscription_plan"].isin(
        sel_plans
    )
    if origin != "All":
        mask &= df["origin_label"] == origin
    df = df[mask].copy()

    if df.empty:
        st.warning("No data matches current filters.")
        return

    uc2.render_kpis(df, title_df)
    st.divider()
    uc2.render_quality_origin(df)
    st.divider()
    uc2.render_subscriber_health(df)
    st.divider()
    uc2.render_content_yield(title_df)
    st.divider()
    uc2.render_genre_origin(df, title_df)


if __name__ == "__main__":
    run_usecase_2()
