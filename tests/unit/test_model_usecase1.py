from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from churn_model_updated import (
    create_segments,
    load_data,
    prepare_features,
    train_model,
)


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


