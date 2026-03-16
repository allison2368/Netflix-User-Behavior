"""
Unit Tests for usecase1.py
Tests for core business logic (no UI tests)

Run: pytest test_usecase1.py -v

"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, MagicMock
import sys

# Mock streamlit and other dependencies
sys.modules['streamlit'] = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()

from usecase1 import (
    predict_churn, 
    apply_filters_and_calculate, 
    SUBSCRIPTION_PLANS,
    PROJECT_ID,
    MODEL_DIR
)


# fixtures for test data a

@pytest.fixture
def sample_data():
    """Sample user data."""
    return pd.DataFrame({
        'user_id': ['U001', 'U002', 'U003'],
        'days_since_last_watch': [5, 15, 25],
        'total_sessions': [100, 80, 60],
        'avg_completion_rate': [0.9, 0.8, 0.7],
        'subscription_plan': ['Premium', 'Standard', 'Basic']
    })


@pytest.fixture
def mock_model():
    """Mock model artifacts."""
    return {
        'rf_model': Mock(predict_proba=Mock(
            return_value=np.array([[0.85, 0.15], [0.75, 0.25], [0.90, 0.10]])
        )),
        'scaler': Mock(transform=Mock(return_value=np.random.randn(3, 2))),
        'rfe_scaler': Mock(transform=Mock(return_value=np.random.randn(3, 3))),
        'kmeans': Mock(predict=Mock(return_value=np.array([0, 1, 2]))),
        'feature_cols': ['days_since_last_watch', 'total_sessions'],
        'config': {
            'N_CLUSTERS': 4,
            'CHURN_THRESHOLD': 0.2,
            'PREDICTION_THRESHOLD': 0.3
        },
        'segment_profile': pd.DataFrame({
            'days_since_last_watch': [10, 20, 30, 40],
            'total_sessions': [100, 80, 60, 40],
            'avg_completion_rate': [0.9, 0.7, 0.5, 0.3]
        })
    }


@pytest.fixture
def filtered_data():
    """Data with predictions."""
    return pd.DataFrame({
        'user_id': ['U001', 'U002', 'U003'],
        'churn_probability_pct': [90.0, 75.0, 95.0],
        'segment': [0, 1, 2],
        'subscription_plan': ['Premium', 'Standard', 'Basic'],
        'days_since_last_watch': [5, 15, 25],
        'avg_completion_rate': [0.9, 0.8, 0.7],
        'monthly_revenue': [19.99, 15.49, 9.99]
    })


# configuration tests

def test_subscription_plans_exist():
    """Test subscription plans are configured."""
    assert 'Basic' in SUBSCRIPTION_PLANS
    assert 'Standard' in SUBSCRIPTION_PLANS
    assert 'Premium' in SUBSCRIPTION_PLANS


def test_subscription_prices_valid():
    """Test prices are positive."""
    for price in SUBSCRIPTION_PLANS.values():
        assert price > 0


def test_project_id_set():
    """Test PROJECT_ID is configured."""
    assert PROJECT_ID == 'netflix-user-behavior'


def test_model_dir_set():
    """Test MODEL_DIR is configured."""
    assert MODEL_DIR == './model_outputs'


# prediction churn tests

def test_predict_churn_adds_columns(sample_data, mock_model):
    """Test prediction adds required columns."""
    result = predict_churn(sample_data, mock_model)
    assert 'segment' in result.columns
    assert 'churn_probability_pct' in result.columns


def test_predict_churn_probability_range(sample_data, mock_model):
    """Test probabilities are 0-100."""
    result = predict_churn(sample_data, mock_model)
    assert result['churn_probability_pct'].min() >= 0
    assert result['churn_probability_pct'].max() <= 100


def test_predict_churn_handles_nan(sample_data, mock_model):
    """Test handles NaN values."""
    sample_data.loc[0, 'total_sessions'] = np.nan
    result = predict_churn(sample_data, mock_model)
    assert 'churn_probability_pct' in result.columns


def test_predict_churn_handles_infinity(sample_data, mock_model):
    """Test handles infinity values."""
    # Convert to float first to avoid dtype warning
    sample_data['total_sessions'] = sample_data['total_sessions'].astype(float)
    sample_data.loc[0, 'total_sessions'] = np.inf
    result = predict_churn(sample_data, mock_model)
    assert np.isfinite(result['churn_probability_pct']).all()


def test_predict_churn_preserves_columns(sample_data, mock_model):
    """Test original columns are preserved."""
    result = predict_churn(sample_data, mock_model)
    for col in sample_data.columns:
        assert col in result.columns


def test_predict_churn_with_extra_columns(sample_data, mock_model):
    """Test handles extra non-numeric columns."""
    sample_data['extra_text'] = ['A', 'B', 'C']
    result = predict_churn(sample_data, mock_model)
    assert 'churn_probability_pct' in result.columns


# filter tests

def test_filter_by_threshold(filtered_data):
    """Test churn threshold filter."""
    filters = {
        'churn_threshold': 85,
        'segments': ['Segment 0', 'Segment 1', 'Segment 2'],
        'plans': ['Premium', 'Standard', 'Basic']
    }
    result, metrics = apply_filters_and_calculate(filtered_data, filters)
    
    assert len(result) == 2
    assert all(result['churn_probability_pct'] >= 85)


def test_filter_by_segment(filtered_data):
    """Test segment filter."""
    filters = {
        'churn_threshold': 70,
        'segments': ['Segment 0'],
        'plans': ['Premium', 'Standard', 'Basic']
    }
    result, metrics = apply_filters_and_calculate(filtered_data, filters)
    
    assert all(result['segment'] == 0)


def test_filter_by_plan(filtered_data):
    """Test subscription plan filter."""
    filters = {
        'churn_threshold': 70,
        'segments': ['Segment 0', 'Segment 1', 'Segment 2'],
        'plans': ['Premium']
    }
    result, metrics = apply_filters_and_calculate(filtered_data, filters)
    
    assert all(result['subscription_plan'] == 'Premium')


def test_filter_combined(filtered_data):
    """Test multiple filters together."""
    filters = {
        'churn_threshold': 90,
        'segments': ['Segment 0', 'Segment 2'],
        'plans': ['Premium', 'Basic']
    }
    result, metrics = apply_filters_and_calculate(filtered_data, filters)
    
    assert all(result['churn_probability_pct'] >= 90)
    assert all(result['segment'].isin([0, 2]))


def test_filter_empty_result(filtered_data):
    """Test when no users match filters."""
    filters = {
        'churn_threshold': 99,
        'segments': ['Segment 0'],
        'plans': ['Basic']
    }
    result, metrics = apply_filters_and_calculate(filtered_data, filters)
    
    assert len(result) == 0
    assert metrics['num_high_risk'] == 0
    assert metrics['avg_churn_prob'] == 0


def test_filter_all_segments(filtered_data):
    """Test selecting all segments."""
    filters = {
        'churn_threshold': 70,
        'segments': ['Segment 0', 'Segment 1', 'Segment 2'],
        'plans': ['Premium', 'Standard', 'Basic']
    }
    result, metrics = apply_filters_and_calculate(filtered_data, filters)
    
    assert len(result) == 3


# metric tests

def test_metrics_calculation(filtered_data):
    """Test metrics are calculated correctly."""
    filters = {
        'churn_threshold': 85,
        'segments': ['Segment 0', 'Segment 1', 'Segment 2'],
        'plans': ['Premium', 'Standard', 'Basic']
    }
    result, metrics = apply_filters_and_calculate(filtered_data, filters)
    
    assert metrics['num_high_risk'] == len(result)
    assert 'avg_churn_prob' in metrics
    assert 'total_revenue' in metrics


def test_metrics_revenue_with_plans(filtered_data):
    """Test revenue calculation with subscription plans."""
    filters = {
        'churn_threshold': 70,
        'segments': ['Segment 0', 'Segment 1', 'Segment 2'],
        'plans': ['Premium', 'Standard', 'Basic']
    }
    result, metrics = apply_filters_and_calculate(filtered_data, filters)
    
    expected_revenue = result['subscription_plan'].map(SUBSCRIPTION_PLANS).sum()
    assert abs(metrics['total_revenue'] - expected_revenue) < 0.01


def test_metrics_revenue_without_plans():
    """Test revenue calculation without subscription plan column."""
    data = pd.DataFrame({
        'user_id': ['U001', 'U002'],
        'churn_probability_pct': [90.0, 85.0],
        'segment': [0, 1]
    })
    
    filters = {
        'churn_threshold': 80,
        'segments': ['Segment 0', 'Segment 1'],
        'plans': ['Basic', 'Standard', 'Premium']
    }
    
    result, metrics = apply_filters_and_calculate(data, filters)
    
    avg_price = np.mean(list(SUBSCRIPTION_PLANS.values()))
    expected = len(result) * avg_price
    assert abs(metrics['total_revenue'] - expected) < 0.01


def test_metrics_average_churn_prob(filtered_data):
    """Test average churn probability calculation."""
    filters = {
        'churn_threshold': 70,
        'segments': ['Segment 0', 'Segment 1', 'Segment 2'],
        'plans': ['Premium', 'Standard', 'Basic']
    }
    result, metrics = apply_filters_and_calculate(filtered_data, filters)
    
    expected_avg = result['churn_probability_pct'].mean()
    assert abs(metrics['avg_churn_prob'] - expected_avg) < 0.01


# edge case tests

def test_single_user(mock_model):
    """Test with single user."""
    data = pd.DataFrame({
        'user_id': ['U001'],
        'days_since_last_watch': [5],
        'total_sessions': [100],
        'avg_completion_rate': [0.9]
    })
    
    mock_model['rf_model'].predict_proba.return_value = np.array([[0.9, 0.1]])
    mock_model['rfe_scaler'].transform.return_value = np.array([[1, 2, 3]])
    mock_model['kmeans'].predict.return_value = np.array([0])
    mock_model['scaler'].transform.return_value = np.array([[1, 2]])
    
    result = predict_churn(data, mock_model)
    assert len(result) == 1


def test_large_dataset(mock_model):
    """Test with larger dataset."""
    n = 100
    data = pd.DataFrame({
        'user_id': [f'U{i:03d}' for i in range(n)],
        'days_since_last_watch': np.random.randint(0, 60, n),
        'total_sessions': np.random.randint(1, 200, n),
        'avg_completion_rate': np.random.uniform(0.1, 1.0, n)
    })
    
    mock_model['rf_model'].predict_proba.return_value = np.random.rand(n, 2)
    mock_model['rfe_scaler'].transform.return_value = np.random.randn(n, 3)
    mock_model['kmeans'].predict.return_value = np.random.randint(0, 4, n)
    mock_model['scaler'].transform.return_value = np.random.randn(n, 2)
    
    result = predict_churn(data, mock_model)
    assert len(result) == n


def test_all_same_segment():
    """Test when all users in same segment."""
    data = pd.DataFrame({
        'user_id': ['U001', 'U002', 'U003'],
        'churn_probability_pct': [90.0, 85.0, 95.0],
        'segment': [0, 0, 0],
        'subscription_plan': ['Premium', 'Standard', 'Basic']
    })
    
    filters = {
        'churn_threshold': 80,
        'segments': ['Segment 0'],
        'plans': ['Premium', 'Standard', 'Basic']
    }
    
    result, metrics = apply_filters_and_calculate(data, filters)
    assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
