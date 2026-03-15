import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import pytest
from churn_model.churn_model_updated import prepare_features, create_segments, train_model, load_data


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

def test_train_model_single_class():
    """
    Verify that the training pipeline does not crash when all samples
    belong to a single target class.

    This simulates a dataset where all users are either churners or
    non-churners, which can cause failures in stratified splitting
    or resampling steps.
    """

    n = 50

    df = pd.DataFrame({
        "user_id": np.arange(n),
        "watch_decline_ratio": np.zeros(n),
        "watch_last_7d": np.random.rand(n),
        "total_watch_minutes": np.random.rand(n),
        "is_active": np.ones(n),
        "feature1": np.random.rand(n)
    })

    result = train_model(df)

    assert result["model"] is not None


def test_train_model_high_class_imbalance():
    """
    Ensure the training pipeline handles extreme class imbalance.

    This test creates a dataset where only a small fraction of users
    belong to the minority churn class, verifying that the model can
    still train and return evaluation metrics.
    """

    n = 200

    y = np.zeros(n)
    y[:5] = 1

    df = pd.DataFrame({
        "user_id": np.arange(n),
        "watch_decline_ratio": y,
        "watch_last_7d": np.random.rand(n),
        "total_watch_minutes": np.random.rand(n),
        "is_active": np.random.randint(0, 2, n),
        "feature1": np.random.rand(n)
    })

    result = train_model(df)

    assert "metrics" in result


# --------------------------------------------------
# BIGQUERY EDGE CASE TESTS
# --------------------------------------------------

@patch("churn_model.churn_model_updated.bigquery.Client")
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


@patch("churn_model.churn_model_updated.bigquery.Client")
def test_load_data_query_failure(mock_client):
    """Ensure load_data propagates BigQuery errors."""

    mock_client.return_value.query.side_effect = Exception("BigQuery error")

    with pytest.raises(Exception):
        load_data()
    
@patch("churn_model.churn_model_updated.bigquery.Client")
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
    
	#--------------------------------------------------
    # add more tests for code coverage
    #--------------------------------------------------
    
def test_load_data_fills_na():
    """Test that load_data fills missing values with 0."""
    import pandas as pd
    from unittest.mock import patch, MagicMock
    from churn_model.churn_model_updated import load_data

    mock_df = pd.DataFrame({
        "a": [1, None],
        "b": [None, 2]
    })

    mock_job = MagicMock()
    mock_job.result.return_value.to_dataframe.return_value = mock_df

    with patch("churn_model.churn_model_updated.bigquery.Client") as mock_client:
        mock_client.return_value.query.return_value = mock_job

        df = load_data()

    assert df.isna().sum().sum() == 0
    
def test_create_segments_adds_segment():
    """Ensure segmentation adds cluster column."""
    import pandas as pd
    from churn_model.churn_model_updated import create_segments

    df = pd.DataFrame({
        "days_since_last_watch":[1,2,3,4],
        "total_sessions":[5,6,7,8],
        "avg_completion_rate":[0.3,0.5,0.6,0.8]
    })

    df_out, scaler, kmeans, profile = create_segments(df)

    assert "segment" in df_out.columns
    assert profile.shape[0] > 0
    
def test_save_artifacts_local(tmp_path):
    """Test saving artifacts locally."""
    import pandas as pd
    from churn_model import churn_model_updated as cm

    cm.MODEL_SAVE_DIR = tmp_path
    cm.USE_GCS = False

    result = {
        "model": None,
        "scaler": None,
        "feature_cols": ["a","b"],
        "metrics":{
            "feature_importance": pd.DataFrame({
                "feature":["a","b"],
                "importance":[0.6,0.4]
            })
        }
    }

    scaler_rfe = None
    kmeans = None
    segment_profile = pd.DataFrame({"x":[1,2]})

    cm.save_artifacts(result, scaler_rfe, kmeans, segment_profile)

    assert (tmp_path / "rf_model.pkl").exists()
    
def test_main_pipeline_runs():
    """Test full pipeline execution."""
    from unittest.mock import patch, MagicMock
    import pandas as pd
    from churn_model import churn_model_updated as cm

    fake_df = pd.DataFrame({
        "user_id":[1,2,3,4],
        "watch_decline_ratio":[0.1,0.5,0.3,0.6],
        "watch_last_7d":[1,1,1,1],
        "total_watch_minutes":[10,20,30,40],
        "is_active":[1,1,1,1],
        "days_since_last_watch":[1,2,3,4],
        "total_sessions":[1,2,3,4],
        "avg_completion_rate":[0.5,0.6,0.7,0.8]
    })

    with patch.object(cm, "load_data", return_value=fake_df), \
         patch.object(cm, "save_artifacts"), \
         patch.object(cm, "create_segments") as mock_seg:

        mock_seg.return_value = (fake_df, None, None, fake_df)

        cm.main()




