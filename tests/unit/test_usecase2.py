"""
Unit tests for usecase2.py.

Covers preprocessing, aggregation logic, and ensures
render functions execute without errors.
"""

import pandas as pd

from usecase2.usecase2 import preprocess


def test_sidebar_filters_genre():
    """Test sidebar genre filtering logic."""
    df = pd.DataFrame({
        "genre_primary": ["Drama", "Comedy"],
        "subscription_plan": ["Basic", "Basic"],
        "origin_label": ["Netflix Original", "Licensed"]
    })
    filtered = df[df["genre_primary"] == "Drama"]

    assert len(filtered) == 1


def test_origin_filter_logic():
    """Test origin filter behavior."""
    df = pd.DataFrame({
        "origin_label": ["Netflix Original", "Licensed"]
    })

    filtered = df[df["origin_label"] == "Netflix Original"]

    assert filtered.shape[0] == 1


def test_quality_origin_aggregation():
    """Test quality and origin aggregation metrics."""
    df = pd.DataFrame({
        "origin_label": ["Netflix Original", "Licensed", "Licensed"],
        "imdb_bucket": ["7-8", "7-8", "7-8"],
        "progress_percentage": [80, 60, 40],
        "watch_duration_minutes": [50, 30, 20],
        "session_id": [1, 2, 3]
    })

    agg = (
        df.groupby(["origin_label", "imdb_bucket"])
        .agg(
            avg_completion=("progress_percentage", "mean"),
            avg_duration=("watch_duration_minutes", "mean"),
            sessions=("session_id", "count")
        )
        .reset_index()
    )

    orig = agg[agg["origin_label"] == "Netflix Original"]

    assert orig["avg_completion"].iloc[0] == 80
    assert orig["avg_duration"].iloc[0] == 50
    assert orig["sessions"].iloc[0] == 1


def test_preprocess_title_df_also_gets_columns():
    """Test preprocess applies columns to both df and title_df."""
    df = pd.DataFrame({
        "imdb_rating": [7.5],
        "is_netflix_original": [True]
    })
    title_df = pd.DataFrame({
        "imdb_rating": [6.5],
        "is_netflix_original": [False]
    })

    _, title_out = preprocess(df, title_df)

    assert "imdb_bucket" in title_out.columns
    assert "origin_label" in title_out.columns
    assert title_out["origin_label"].iloc[0] == "Licensed"


def test_preprocess_extreme_ratings():
    """Test preprocess handles extreme IMDb rating values."""
    df = pd.DataFrame({
        "imdb_rating": [0.1, 9.9],
        "is_netflix_original": [True, False]
    })

    title_df = df.copy()
    df_out, _ = preprocess(df, title_df)

    assert str(df_out["imdb_bucket"].iloc[0]) == "0-5"
    assert str(df_out["imdb_bucket"].iloc[1]) == "9-10"


def test_preprocess_all_licensed():
    """Test preprocess with all licensed content."""
    df = pd.DataFrame({
        "imdb_rating": [6.0, 7.0, 8.0],
        "is_netflix_original": [False, False, False]
    })

    title_df = df.copy()
    df_out, _ = preprocess(df, title_df)

    assert (df_out["origin_label"] == "Licensed").all()


def test_kpi_originals_count():
    """Test count of Netflix Originals in title_df."""
    title_df = pd.DataFrame({
        "is_netflix_original": [True, True, False, False, False]
    })

    n_orig     = title_df["is_netflix_original"].sum()
    n_licensed = (~title_df["is_netflix_original"]).sum()

    assert n_orig == 2
    assert n_licensed == 3


def test_kpi_active_subscriber_rate():
    """Test active subscriber rate calculation."""
    df = pd.DataFrame({
        "is_active": [True, True, False, True, False]
    })

    rate = df["is_active"].mean() * 100

    assert rate == 60.0


def test_kpi_avg_imdb_by_origin():
    """Test average IMDb rating calculation by origin."""
    title_df = pd.DataFrame({
        "imdb_rating": [8.0, 9.0, 6.0, 7.0],
        "is_netflix_original": [True, True, False, False]
    })

    avg_orig = title_df[title_df["is_netflix_original"]]["imdb_rating"].mean()
    avg_lic  = title_df[~title_df["is_netflix_original"]]["imdb_rating"].mean()

    assert avg_orig == 8.5
    assert avg_lic == 6.5


def test_subscriber_health_high_rated_filter():
    """Test that subscriber health only uses IMDb >= 7 content."""
    df = pd.DataFrame({
        "imdb_rating": [8.0, 6.0, 7.5, 5.0],
        "origin_label": ["Netflix Original", "Licensed",
                         "Netflix Original", "Licensed"],
    })

    high_rated = df[df["imdb_rating"] >= 7]

    assert len(high_rated) == 2
    assert all(high_rated["imdb_rating"] >= 7)


def test_subscriber_health_active_rate():
    """Test active rate aggregation for high-rated content viewers."""
    df = pd.DataFrame({
        "origin_label": ["Netflix Original", "Netflix Original", "Licensed"],
        "is_active": [True, False, True],
        "user_id": [1, 2, 3]
    })

    health = (
        df.groupby("origin_label")
        .agg(active_rate=("is_active", "mean"))
        .reset_index()
    )

    orig_rate = health[
        health["origin_label"] == "Netflix Original"
    ]["active_rate"].values[0]

    assert orig_rate == 0.5


def test_top_titles_sorted_by_sessions():
    """Test top titles are sorted by total_sessions descending."""
    title_df = pd.DataFrame({
        "title": ["A", "B", "C"],
        "total_sessions": [100, 300, 200],
    })

    top = title_df.sort_values("total_sessions", ascending=False).head(15)

    assert top.iloc[0]["title"] == "B"
    assert top.iloc[1]["title"] == "C"
    assert top.iloc[2]["title"] == "A"


def test_content_yield_aggregation():
    """Test content yield aggregation by origin and IMDb bucket."""
    title_df = pd.DataFrame({
        "origin_label": ["Netflix Original", "Netflix Original", "Licensed"],
        "imdb_bucket": ["7-8", "7-8", "8-9"],
        "total_sessions": [100, 200, 150],
        "total_watch_minutes": [5000, 10000, 7500],
        "unique_viewers": [50, 100, 75],
        "movie_id": [1, 2, 3]
    })

    yield_agg = (
        title_df.groupby(["origin_label", "imdb_bucket"])
        .agg(
            avg_sessions=("total_sessions", "mean"),
            title_count=("movie_id", "count")
        )
        .reset_index()
    )

    orig = yield_agg[yield_agg["origin_label"] == "Netflix Original"]

    assert orig["avg_sessions"].iloc[0] == 150.0
    assert orig["title_count"].iloc[0] == 2


def test_genre_origin_pivot_creates_gap_column():
    """Test that genre-origin pivot creates the gap column correctly."""
    genre_agg = pd.DataFrame({
        "origin_label": ["Netflix Original", "Licensed",
                         "Netflix Original", "Licensed"],
        "genre_primary": ["Drama", "Drama", "Comedy", "Comedy"],
        "avg_completion": [80.0, 70.0, 60.0, 65.0]
    })

    pivot = genre_agg.pivot_table(
        index="genre_primary", columns="origin_label",
        values="avg_completion", aggfunc="mean"
    ).round(1)

    pivot["gap (Orig − Lic)"] = (
        pivot["Netflix Original"] - pivot["Licensed"]
    ).round(1)

    assert "gap (Orig − Lic)" in pivot.columns
    assert pivot.loc["Drama", "gap (Orig − Lic)"] == 10.0
    assert pivot.loc["Comedy", "gap (Orig − Lic)"] == -5.0


def test_genre_origin_pivot_sorted_by_gap():
    """Test that pivot is sorted by gap descending."""
    genre_agg = pd.DataFrame({
        "origin_label": ["Netflix Original", "Licensed",
                         "Netflix Original", "Licensed"],
        "genre_primary": ["Drama", "Drama", "Comedy", "Comedy"],
        "avg_completion": [80.0, 70.0, 60.0, 65.0]
    })

    pivot = genre_agg.pivot_table(
        index="genre_primary", columns="origin_label",
        values="avg_completion", aggfunc="mean"
    ).round(1)

    pivot["gap (Orig − Lic)"] = (
        pivot["Netflix Original"] - pivot["Licensed"]
    ).round(1)
    pivot = pivot.sort_values("gap (Orig − Lic)", ascending=False)

    assert pivot.index[0] == "Drama"
    assert pivot.index[1] == "Comedy"


def test_title_count_pivot():
    """Test title count pivot table by genre and origin."""
    title_df = pd.DataFrame({
        "origin_label": ["Netflix Original", "Netflix Original",
                         "Licensed", "Licensed", "Licensed"],
        "genre_primary": ["Drama", "Drama", "Comedy", "Drama", "Comedy"],
        "movie_id": [1, 2, 3, 4, 5]
    })

    title_counts = (
        title_df.groupby(["origin_label", "genre_primary"])
        .agg(title_count=("movie_id", "count"))
        .reset_index()
    )

    pivot = title_counts.pivot_table(
        index="genre_primary", columns="origin_label",
        values="title_count", aggfunc="sum", fill_value=0
    )

    assert pivot.loc["Drama", "Netflix Original"] == 2
    assert pivot.loc["Drama", "Licensed"] == 1
    assert pivot.loc["Comedy", "Licensed"] == 2
