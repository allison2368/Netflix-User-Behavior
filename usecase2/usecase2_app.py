# usecase2/usecase2_app.py
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from .usecase2 import load_data, preprocess, ORIGIN_COLORS 

def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar filters to session-level dataframe.

    Args:
        df (pd.DataFrame): Session-level dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    st.sidebar.header("🎛️ Filters")

    genres = sorted(df["genre_primary"].dropna().unique())
    sel_genres = st.sidebar.multiselect("Genre", genres, default=genres)

    plans = sorted(df["subscription_plan"].dropna().unique())
    sel_plans = st.sidebar.multiselect("Subscription plan", plans, default=plans)

    origin = st.sidebar.radio("Content origin", ["All", "Netflix Original", "Licensed"])

    mask = (
        df["genre_primary"].isin(sel_genres)
        & df["subscription_plan"].isin(sel_plans)
    )
    if origin != "All":
        mask &= df["origin_label"] == origin

    return df[mask].copy()


def render_kpis(df: pd.DataFrame, title_df: pd.DataFrame) -> None:
    """Render top-level KPIs for content portfolio in Streamlit.

    Args:
        df (pd.DataFrame): Session-level dataframe
        title_df (pd.DataFrame): Title-level dataframe
    """
    st.subheader("📊 Portfolio Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    n_orig     = title_df["is_netflix_original"].sum()
    n_licensed = (~title_df["is_netflix_original"]).sum()
    c1.metric("Total titles",         f"{len(title_df):,}")
    c2.metric("Netflix Originals",    f"{n_orig:,}")
    c3.metric("Licensed titles",      f"{n_licensed:,}")
    c4.metric(
        "Avg IMDb (Originals)",
        f"{title_df[title_df.is_netflix_original]['imdb_rating'].mean():.2f}"
    )
    c5.metric(
        "Avg IMDb (Licensed)",
        f"{title_df[~title_df.is_netflix_original]['imdb_rating'].mean():.2f}"
    )
    c6.metric("Active subscriber %",  f"{df['is_active'].mean() * 100:.1f}%")


# Analysis 1
def render_quality_origin(df: pd.DataFrame) -> None:
    """Render quality tier × origin type analysis (completion and watch duration).

    Args:
        df (pd.DataFrame): Session-level dataframe
    """
    st.subheader("1️⃣ Quality Tier × Origin Type → Retention")
    st.caption("Does high-rated content drive more engagement, "
               "and does origin type amplify that effect?")

    agg = (
        df.groupby(["origin_label", "imdb_bucket"], observed=True)
        .agg(
            avg_completion = ("progress_percentage",    "mean"),
            avg_duration   = ("watch_duration_minutes", "mean"),
            sessions       = ("session_id",             "count"),
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)

    # Line: completion by IMDb bucket, split by origin
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for origin, color in ORIGIN_COLORS.items():
            sub = agg[agg["origin_label"] == origin]
            ax.plot(sub["imdb_bucket"].astype(str), sub["avg_completion"],
                    marker="o", linewidth=2.5, markersize=7,
                    color=color, label=origin)
        ax.set_xlabel("IMDb Rating Bucket")
        ax.set_ylabel("Avg Completion Rate (%)")
        ax.set_title("Completion Rate by Quality Tier & Origin", fontsize=11, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

    # Line: watch duration by IMDb bucket, split by origin
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        for origin, color in ORIGIN_COLORS.items():
            sub = agg[agg["origin_label"] == origin]
            ax.plot(sub["imdb_bucket"].astype(str), sub["avg_duration"],
                    marker="o", linewidth=2.5, markersize=7,
                    color=color, label=origin)
        ax.set_xlabel("IMDb Rating Bucket")
        ax.set_ylabel("Avg Watch Duration (min)")
        ax.set_title("Watch Duration by Quality Tier & Origin", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

    # Heatmap: imdb_bucket × origin = avg completion
    pivot = agg.pivot_table(
        index="imdb_bucket", columns="origin_label",
        values="avg_completion", aggfunc="mean",
    ).round(1)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f", cmap="YlGnBu",
                linewidths=0.4, cbar_kws={"label": "Avg Completion %"})
    ax.set_title("Completion % Heatmap — Quality × Origin", fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("IMDb Bucket")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("📋 Full table"):
        st.dataframe(agg.round(2), use_container_width=True)

# Analysis 2
def _render_kpi_cards(health: pd.DataFrame) -> None:
    """Render KPI cards horizontally for all origins."""
    for _, row in health.iterrows():
        color = TEAL if row["origin_label"] == "Netflix Original" else AMBER
        st.markdown(f"<h4 style='color:{color}'>{row['origin_label']}</h4>", unsafe_allow_html=True)
        cols = st.columns(6)
        cols[0].metric("Active rate", f"{row['active_rate']*100:.1f}%")
        cols[1].metric("Avg spend/mo", f"${row['avg_monthly_spend']:.2f}")
        cols[2].metric("Avg tenure", f"{row['avg_tenure_days']:.0f}d")
        cols[3].metric("Watch decline", f"{row['avg_watch_decline']:.2f}")
        cols[4].metric("7v30 ratio", f"{row['avg_engagement_ratio']:.2f}")
        cols[5].metric("Watch last 30d", f"{row['avg_watch_last30']:.0f} min")


def _plot_health_bar_chart(health: pd.DataFrame) -> None:
    """Render bar chart for subscriber health metrics."""
    metrics = ["active_rate", "avg_engagement_ratio", "avg_watch_decline"]
    labels  = ["Active Rate", "Engagement 7v30", "Watch Decline Ratio"]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics))
    width = 0.35

    for i, (origin, color) in enumerate(ORIGIN_COLORS.items()):
        subset = health[health["origin_label"] == origin]
        if subset.empty:
            continue
        heights = [subset[m].values[0] for m in metrics]
        ax.bar(x + i*width, heights, width=width, color=color, alpha=0.85, label=origin)
        for xi, h in zip(x + i*width, heights):
            ax.text(xi, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width/2)
    ax.set_xticklabels(labels)
    ax.set_title("Subscriber Health — High-Rated Content Viewers (IMDb ≥ 7)")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

def _plot_active_rate_by_bucket(df: pd.DataFrame) -> None:
    """Render active subscriber rate by IMDb bucket and origin."""
    high_rated = df[df["imdb_rating"] >= 7].copy()
    active_agg = (
        high_rated.groupby(["origin_label", "imdb_bucket"], observed=True)
        .agg(active_rate=("is_active", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    for origin, color in ORIGIN_COLORS.items():
        sub = active_agg[active_agg["origin_label"] == origin]
        ax.plot(
            sub["imdb_bucket"].astype(str),
            sub["active_rate"] * 100,
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=color,
            label=origin,
        )
    ax.set_xlabel("IMDb Rating Bucket")
    ax.set_ylabel("Active Subscriber Rate (%)")
    ax.set_title("Active Rate by Quality Tier & Origin", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)


def render_subscriber_health(df: pd.DataFrame) -> None:
    """Main function to render subscriber health analysis."""
    st.subheader("2️⃣ High-Rated Originals vs Licensed → Subscriber Health")
    st.caption(
        "Among users who watch high-rated content (IMDb ≥ 7), "
        "do Netflix Originals produce healthier subscribers?"
    )

    high_rated = df[df["imdb_rating"] >= 7].copy()

    health = (
        high_rated.groupby("origin_label")
        .agg(
            users=("user_id", "nunique"),
            active_rate=("is_active", "mean"),
            avg_monthly_spend=("monthly_spend", "mean"),
            avg_tenure_days=("tenure_days", "mean"),
            avg_watch_decline=("watch_decline_ratio", "mean"),
            avg_engagement_ratio=("engagement_ratio_7v30", "mean"),
            avg_watch_last30=("watch_last_30d", "mean"),
        )
        .reset_index()
        .round(3)
    )

    _render_kpi_cards(health)
    st.divider()

    _plot_health_bar_chart(health)

    _plot_active_rate_by_bucket(df)

# Analysis 3
def _plot_content_yield_box(title_df: pd.DataFrame) -> None:
    """Boxplot: sessions per title by origin.

    Args:
        title_df (pd.DataFrame): Title-level dataframe
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    data_orig = title_df[title_df["origin_label"] == "Netflix Original"]["total_sessions"]
    data_lic = title_df[title_df["origin_label"] == "Licensed"]["total_sessions"]

    bp = ax.boxplot([data_orig, data_lic], labels=["Netflix Original", "Licensed"],
                    patch_artist=True, medianprops={"color": "white", "linewidth": 2})
    bp["boxes"][0].set_facecolor(TEAL)
    bp["boxes"][1].set_facecolor(AMBER)
    for patch in bp["boxes"]:
        patch.set_alpha(0.75)

    ax.set_ylabel("Total Sessions per Title")
    ax.set_title("Session Yield per Title by Origin", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


def _plot_content_yield_scatter(title_df: pd.DataFrame) -> None:
    """Scatterplot: IMDb rating vs total sessions by origin.

    Args:
        title_df (pd.DataFrame): Title-level dataframe
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for origin, color in ORIGIN_COLORS.items():
        sub = title_df[title_df["origin_label"] == origin]
        ax.scatter(sub["imdb_rating"],
                   sub["total_sessions"],
                   alpha=0.4,
                   s=20,
                   color=color,
                   label=origin
        )
    ax.set_xlabel("IMDb Rating")
    ax.set_ylabel("Total Sessions")
    ax.set_title("IMDb Rating vs Session Yield per Title", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)


def _plot_content_yield_lines(title_df: pd.DataFrame) -> None:
    """Line plots for avg sessions and watch minutes by origin and IMDb bucket.

    Args:
        title_df (pd.DataFrame): Title-level dataframe
    """
    yield_agg = title_df.groupby(["origin_label", "imdb_bucket"], observed=True).agg(
        avg_sessions=("total_sessions", "mean"),
        avg_watch_minutes=("total_watch_minutes", "mean"),
        avg_viewers=("unique_viewers", "mean"),
        title_count=("movie_id", "count"),
    ).reset_index().round(1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, metric, label in zip(
        axes,
        ["avg_sessions", "avg_watch_minutes"],
        ["Avg Sessions per Title", "Avg Total Watch Minutes per Title"]
    ):
        for origin, color in ORIGIN_COLORS.items():
            sub = yield_agg[yield_agg["origin_label"] == origin]
            ax.plot(sub["imdb_bucket"].astype(str), sub[metric],
                    marker="o", linewidth=2.5, markersize=7, color=color, label=origin)
        ax.set_xlabel("IMDb Rating Bucket")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by Quality Tier", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)


def render_content_yield(title_df: pd.DataFrame) -> None:
    """Main function to render content yield analysis."""
    st.subheader("3️⃣ Content Yield per Title")
    st.caption(
        "Which titles punch above their weight? Sessions and watch minutes per title, "
        "by origin and quality."
    )

    col1, col2 = st.columns(2)
    with col1:
        _plot_content_yield_box(title_df)
    with col2:
        _plot_content_yield_scatter(title_df)

    _plot_content_yield_lines(title_df)

    st.markdown("**Top 15 Titles by Total Sessions**")
    top_titles = title_df.sort_values("total_sessions", ascending=False).head(15)[
        ["title", "origin_label", "imdb_rating", "genre_primary",
         "total_sessions", "unique_viewers", "avg_completion"]
    ].round(2).reset_index(drop=True)
    st.dataframe(top_titles, use_container_width=True)

# Analysis 4
def render_genre_origin(df: pd.DataFrame, title_df: pd.DataFrame) -> None:
    """Render genre-level analysis comparing Netflix Originals vs Licensed content.

    Args:
        df (pd.DataFrame): Session-level dataframe
        title_df (pd.DataFrame): Title-level dataframe
    """
    st.subheader("4️⃣ Genre Breakdown by Origin Type")
    st.caption("Where do Netflix Originals outperform licensed content at the genre level?")

    genre_agg = (
        df.groupby(["origin_label", "genre_primary"], observed=True)
        .agg(
            avg_completion = ("progress_percentage",    "mean"),
            avg_duration   = ("watch_duration_minutes", "mean"),
            sessions       = ("session_id",             "count"),
        )
        .reset_index()
        .round(2)
    )

    pivot_completion = genre_agg.pivot_table(
        index="genre_primary", columns="origin_label",
        values="avg_completion", aggfunc="mean",
    ).round(1)

    if "Netflix Original" in pivot_completion.columns and "Licensed" in pivot_completion.columns:
        pivot_completion["gap (Orig − Lic)"] = (
            pivot_completion["Netflix Original"] - pivot_completion["Licensed"]
        ).round(1)
        pivot_completion = pivot_completion.sort_values("gap (Orig − Lic)", ascending=False)

    fig, ax = plt.subplots(figsize=(8, max(5, len(pivot_completion) * 0.45)))
    sns.heatmap(
        pivot_completion, ax=ax, annot=True, fmt=".1f",
        cmap="RdYlGn", center=0,
        linewidths=0.4, cbar_kws={"label": "Avg Completion %"},
    )
    ax.set_title("Completion % by Genre × Origin  (gap = Original − Licensed)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Genre")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    st.pyplot(fig)
    plt.close(fig)

    title_counts = (
        title_df.groupby(["origin_label", "genre_primary"], observed=True)
        .agg(title_count=("movie_id", "count"))
        .reset_index()
    )
    pivot_count = title_counts.pivot_table(
        index="genre_primary", columns="origin_label",
        values="title_count", aggfunc="sum", fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(8, max(5, len(pivot_count) * 0.45)))
    sns.heatmap(pivot_count, ax=ax, annot=True, fmt=".0f", cmap="Blues",
                linewidths=0.4, cbar_kws={"label": "Number of Titles"})
    ax.set_title("Title Count by Genre × Origin  (volume vs quality trade-off)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("📋 Full genre breakdown"):
        st.dataframe(genre_agg, use_container_width=True)

def run_usecase2() -> None:
    """Run the Content Investment Optimization dashboard."""
    st.title("🎬 Content Investment Optimization")
    st.markdown(
        "**Sarah's question:** *Does investing in fewer high-rated Netflix Originals "
        "drive more retention than a higher volume of average-rated licensed content?*"
    )
    st.divider()

    df_raw, title_df_raw = load_data()
    df, title_df         = preprocess(df_raw.copy(), title_df_raw.copy())
    df                   = sidebar_filters(df)

    if df.empty:
        st.warning("No data matches current filters.")
        return

    render_kpis(df, title_df)
    st.divider()
    render_quality_origin(df)
    st.divider()
    render_subscriber_health(df)
    st.divider()
    render_content_yield(title_df)
    st.divider()
    render_genre_origin(df, title_df)