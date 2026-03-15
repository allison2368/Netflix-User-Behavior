import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from churn_model_updated import prepare_features, create_segments, train_model, load_data


def test_prepare_features_creates_target():
    df = pd.DataFrame({
        "user_id":[1,2],
        "watch_decline_ratio":[0.1,0.5],
        "watch_last_7d":[5,10],
        "total_watch_minutes":[100,200],
        "is_active":[1,1],
        "feature1":[3,4]
    })

    X, y, _ = prepare_features(df)

    assert len(y) == 2
    assert set(y.unique()).issubset({0,1})


def test_create_segments():
    df = pd.DataFrame({
        "days_since_last_watch": np.random.rand(20),
        "total_sessions": np.random.rand(20),
        "avg_completion_rate": np.random.rand(20)
    })

    df_out, _, _, _ = create_segments(df)

    assert "segment" in df_out.columns


def test_train_model_runs():

    n = 100

    df = pd.DataFrame({
        "user_id": np.arange(n),
        "watch_decline_ratio": np.random.rand(n),
        "watch_last_7d": np.random.rand(n),
        "total_watch_minutes": np.random.rand(n),
        "is_active": np.random.randint(0,2,n),
        "feature1": np.random.rand(n),
        "feature2": np.random.rand(n)
    })

    result = train_model(df)

    assert "metrics" in result

# --------------------------------------------------
# PREPARE_FEATURES EDGE CASE TESTS
# --------------------------------------------------

def test_prepare_features_handles_nan():
    """Ensure NaN values are cleaned during feature preparation."""

    dataframe = pd.DataFrame({
        "user_id": [1, 2],
        "watch_decline_ratio": [0.1, 0.5],
        "watch_last_7d": [5, 10],
        "total_watch_minutes": [100, 200],
        "is_active": [1, 1],
        "feature1": [np.nan, 3],
    })

    features, target, _ = prepare_features(dataframe)

    assert features.isna().sum().sum() == 0
    assert len(target) == 2


def test_prepare_features_handles_inf():
    """Ensure infinite values are replaced during feature preparation."""

    dataframe = pd.DataFrame({
        "user_id": [1, 2],
        "watch_decline_ratio": [0.1, 0.5],
        "watch_last_7d": [5, 10],
        "total_watch_minutes": [100, 200],
        "is_active": [1, 1],
        "feature1": [np.inf, -np.inf],
    })

    features, _, _ = prepare_features(dataframe)

    assert np.isfinite(features.values).all()


def test_prepare_features_non_numeric():
    """Ensure non-numeric features are coerced safely."""

    dataframe = pd.DataFrame({
        "user_id": [1, 2],
        "watch_decline_ratio": [0.1, 0.5],
        "watch_last_7d": [5, 10],
        "total_watch_minutes": [100, 200],
        "is_active": [1, 1],
        "feature1": ["bad", "data"],
    })

    features, _, _ = prepare_features(dataframe)

    assert all(dtype == float for dtype in features.dtypes)


# --------------------------------------------------
# SEGMENTATION EDGE CASE TESTS
# --------------------------------------------------

def test_create_segments_cluster_count():
    """Ensure clustering creates valid segment labels."""

    dataframe = pd.DataFrame({
        "days_since_last_watch": np.random.rand(100),
        "total_sessions": np.random.rand(100),
        "avg_completion_rate": np.random.rand(100),
    })

    segmented_df, _, _, _ = create_segments(dataframe)

    assert "segment" in segmented_df.columns
    assert segmented_df["segment"].nunique() <= 4


def test_create_segments_missing_columns():
    """Ensure missing required segmentation columns raises an error."""

    dataframe = pd.DataFrame({
        "total_sessions": np.random.rand(10),
    })

    with pytest.raises(KeyError):
        create_segments(dataframe)


# --------------------------------------------------
# MODEL TRAINING EDGE CASE TESTS
# --------------------------------------------------

def test_train_model_small_dataset():
    """Ensure model can train on a minimal dataset."""

    dataframe = pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "watch_decline_ratio": [0.1, 0.5, 0.3, 0.9],
        "watch_last_7d": [1, 2, 3, 4],
        "total_watch_minutes": [10, 20, 30, 40],
        "is_active": [1, 0, 1, 0],
        "feature1": [1, 2, 3, 4],
    })

    result = train_model(dataframe)

    assert result["model"] is not None


def test_train_model_metrics_exist():
    """Ensure expected evaluation metrics are returned."""

    sample_size = 50

    dataframe = pd.DataFrame({
        "user_id": np.arange(sample_size),
        "watch_decline_ratio": np.random.rand(sample_size),
        "watch_last_7d": np.random.rand(sample_size),
        "total_watch_minutes": np.random.rand(sample_size),
        "is_active": np.random.randint(0, 2, sample_size),
        "feature1": np.random.rand(sample_size),
    })

    result = train_model(dataframe)

    metrics = result["metrics"]

    assert "accuracy" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics


# --------------------------------------------------
# BIGQUERY EDGE CASE TESTS
# --------------------------------------------------

@patch("churn_model_updated.bigquery.Client")
def test_load_data_fills_na(mock_client):
    """Ensure load_data replaces missing values."""

    fake_dataframe = pd.DataFrame({
        "user_id": [1],
        "watch_decline_ratio": [np.nan],
    })

    mock_job = MagicMock()
    mock_job.result.return_value.to_dataframe.return_value = fake_dataframe
    mock_client.return_value.query.return_value = mock_job

    dataframe = load_data()

    assert dataframe.isna().sum().sum() == 0


@patch("churn_model_updated.bigquery.Client")
def test_load_data_query_failure(mock_client):
    """Ensure load_data propagates BigQuery errors."""

    mock_client.return_value.query.side_effect = Exception("BigQuery error")

    with pytest.raises(Exception):
        load_data()
    
@patch("churn_model_updated.bigquery.Client")
def test_load_data(mock_client):

    fake_df = pd.DataFrame({
        "user_id":[1],
        "watch_decline_ratio":[0.2]
    })

    mock_job = MagicMock()
    mock_job.result.return_value.to_dataframe.return_value = fake_df
    mock_client.return_value.query.return_value = mock_job

    df = load_data()

    assert len(df) == 1




