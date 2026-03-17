"""
Tests for usecase2_app.py.

Focuses on ensuring the app runs with mocked data.
"""

import pandas as pd

from usecase2.usecase2 import preprocess


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
    """Test that origin labels map correctly."""
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
    """Test IMDb bucket assignment."""
    df = pd.DataFrame({
        "imdb_rating": [5.5, 7.4, 8.8],
        "is_netflix_original": [True, True, False]
    })

    title_df = df.copy()

    df_out, _ = preprocess(df, title_df)

    assert str(df_out["imdb_bucket"].iloc[0]) == "5-6"
    assert str(df_out["imdb_bucket"].iloc[1]) == "7-8"
    assert str(df_out["imdb_bucket"].iloc[2]) == "8-9"


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
    """Test dataset containing only one origin."""
    df = pd.DataFrame({
        "imdb_rating": [7.5, 8.0],
        "is_netflix_original": [True, True]
    })

    title_df = df.copy()

    df_out, _ = preprocess(df, title_df)

    assert (df_out["origin_label"] == "Netflix Original").all()
