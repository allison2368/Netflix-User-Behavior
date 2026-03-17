"""
Test suite for usecase1 module.

This module contains unit tests for churn prediction functionality,
including artifact loading, prediction logic, and visualization functions.
"""
import os
import pickle
import tempfile

import numpy as np
import pandas as pd

from usecase1.usecase1 import (
    load_artifacts,
    predict_churn_logic,
    get_churn_dist_plot,
    get_segment_pie_plot
)
import usecase1.usecase1 as usecase1

# Dummy helper classes to mock model and scaler behavior for testing purposes

class DummyScaler:
    """Mock scaler for testing purposes."""

    def transform(self, x_data):
        """Return input data unchanged."""
        return x_data

class DummyModel:
    """Mock model for testing purposes."""

    def predict_proba(self, x_data):
        """Return fixed probability predictions."""
        return np.array([[0.4, 0.6]] * len(x_data))

class DummyKMeans:
    """Mock KMeans clusterer for testing purposes."""

    def predict(self, x_data):
        """Return cluster 0 for all inputs."""
        return np.zeros(len(x_data))


# test artifact loading

def test_load_artifacts():
    """Test that artifacts are correctly loaded from disk."""
    with tempfile.TemporaryDirectory() as tmp:

        files = [
            "rf_model",
            "scaler",
            "rfe_scaler",
            "kmeans",
            "feature_names",
            "config"
        ]

        for name in files:
            with open(os.path.join(tmp, f"{name}.pkl"), "wb") as f:
                pickle.dump({"test": 1}, f)

        df = pd.DataFrame({"a":[1,2]})
        df.to_csv(os.path.join(tmp, "segment_profile.csv"))

        usecase1.MODEL_DIR = tmp

        artifacts = load_artifacts()

        assert "segment_profile" in artifacts
        assert isinstance(artifacts["segment_profile"], pd.DataFrame)

# test churn prediction logic

def test_predict_churn_logic():
    """Test that churn prediction logic produces expected output columns."""
    df = pd.DataFrame({
        "user_id": ["A","B"],
        "days_since_last_watch": [5,20],
        "total_sessions": [10,3],
        "avg_completion_rate": [0.8,0.4],
        "subscription_plan": ["Basic","Premium"]
    })

    artifacts = {
        "rfe_scaler": DummyScaler(),
        "kmeans": DummyKMeans(),
        "scaler": DummyScaler(),
        "rf_model": DummyModel(),
        "feature_cols":[
            "days_since_last_watch",
            "total_sessions",
            "avg_completion_rate"
        ]
    }

    result = predict_churn_logic(df, artifacts)

    assert "segment" in result.columns
    assert "churn_probability_pct" in result.columns
    assert len(result) == 2


# test missing feature handling

def test_missing_feature_handling():
    """Test that prediction handles missing features gracefully."""
    df = pd.DataFrame({
        "user_id":["A"],
        "days_since_last_watch":[10],
        "total_sessions":[4],
        "avg_completion_rate":[0.6]
    })

    artifacts = {
        "rfe_scaler": DummyScaler(),
        "kmeans": DummyKMeans(),
        "scaler": DummyScaler(),
        "rf_model": DummyModel(),
        "feature_cols":[
            "days_since_last_watch",
            "total_sessions",
            "avg_completion_rate",
            "extra_feature"
        ]
    }

    result = predict_churn_logic(df, artifacts)

    assert result["churn_probability_pct"].iloc[0] >= 0


# histogram plot

def test_churn_distribution_plot():
    """Test that churn distribution plot is generated correctly."""
    df = pd.DataFrame({
        "churn_probability_pct":[10,40,70,90]
    })

    fig = get_churn_dist_plot(df)

    assert fig.data[0].type == "histogram"
    assert fig.layout.title.text == "Churn Probability Distribution"


# segment pie plot

def test_segment_pie_plot():
    """Test that segment pie plot is generated correctly."""
    df = pd.DataFrame({
        "segment":[0,0,1,2]
    })

    fig = get_segment_pie_plot(df)

    assert fig.data[0].type == "pie"
    assert len(fig.data[0].labels) > 0
