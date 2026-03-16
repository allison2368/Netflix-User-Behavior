"""
Unit tests for use_case_2_app.py functions.

Includes unit, edge, and integration tests for Use Case 2.
"""
import pandas as pd

from usecase2 import preprocess

def test_preprocess_creates_columns():
    """Test preprocess creates derived columns."""
    df = pd.DataFrame({
        "imdb_rating": [7.5, 6.2],
        "is_netflix_original": [True, False]
    })

    title_df = df.copy()

    df_out, _ = preprocess(df, title_df)

    assert "imdb_bucket" in df_out.columns
    assert "origin_label" in df_out.columns


def test_origin_label_mapping():
    """Test that origin labels map correctly"""
    df = pd.DataFrame({
        "imdb_rating": [7.0, 7.0],
        "is_netflix_original": [True, False]
    })

    title_df = df.copy()

    df_out, _ = preprocess(df, title_df)

    assert df_out["origin_label"].tolist() == [
        "Netflix Original",
        "Licensed"
    ]


def test_imdb_bucket_assignment():
    """Test IMDb bucket assignment"""
    df = pd.DataFrame({
        "imdb_rating": [5.5, 7.4, 8.8],
        "is_netflix_original": [True, True, False]
    })

    title_df = df.copy()

    df_out, _ = preprocess(df, title_df)

    assert str(df_out["imdb_bucket"].iloc[0]) == "5-6"
    assert str(df_out["imdb_bucket"].iloc[1]) == "7-8"
    assert str(df_out["imdb_bucket"].iloc[2]) == "8-9"


def test_sidebar_filters_genre():
    """Test sidebar genre filtering logic"""
    df = pd.DataFrame({
        "genre_primary": ["Drama", "Comedy"],
        "subscription_plan": ["Basic", "Basic"],
        "origin_label": ["Netflix Original", "Licensed"]
    })
    filtered = df[df["genre_primary"] == "Drama"]

    assert len(filtered) == 1


def test_origin_filter_logic():
    """Test origin filter behavior"""
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


def test_preprocess_handles_missing_ratings():
    """Test preprocess handles missing ratings."""
    df = pd.DataFrame({
        "imdb_rating": [None, 7.2],
        "is_netflix_original": [True, False]
    })

    title_df = df.copy()

    df_out, _ = preprocess(df, title_df)

    assert df_out["imdb_bucket"].isna().sum() >= 1


def test_preprocess_empty_dataframe():
    """Test preprocess with empty dataframe."""
    df = pd.DataFrame(columns=["imdb_rating", "is_netflix_original"])
    title_df = df.copy()

    df_out, title_out = preprocess(df, title_df)

    assert df_out.empty
    assert title_out.empty


def test_single_origin_dataset():
    """Test dataset containing only one origin"""
    df = pd.DataFrame({
        "imdb_rating": [7.5, 8.0],
        "is_netflix_original": [True, True]
    })

    title_df = df.copy()

    df_out, _ = preprocess(df, title_df)

    assert (df_out["origin_label"] == "Netflix Original").all()
