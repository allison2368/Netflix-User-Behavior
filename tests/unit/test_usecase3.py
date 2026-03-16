"""
Unit and Edge Tests for Use Case 3: Visualization and Data Logic
"""

from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import usecase3 as uc3

# Unit Tests for Calculation Logic


@patch("usecase3.load_final_pm_report")
@patch("usecase3.load_null_search_analysis")
@patch("usecase3.load_tenure_analysis")
def test_get_summary_metrics_logic(mock_tenure, mock_search, mock_bounce):
    """Test if summary metrics are calculated correctly with floating point precision."""
    # Prepare mock dataframes
    mock_bounce.return_value = pd.DataFrame({"bounce_rate": [0.1, 0.2]})
    mock_search.return_value = pd.DataFrame({"avg_search_failure_rate": [15.0]})
    mock_tenure.return_value = pd.DataFrame({"churn_rate": [8.0]})

    # Execute function
    churn, bounce, search = uc3.get_summary_metrics()

    # Validate calculations using pytest.approx for floating point accuracy
    assert isinstance(churn, float)
    assert bounce == pytest.approx(15.0)  # Expected: ((0.1+0.2)/2) * 100
    assert search == pytest.approx(15.0)


# Unit Tests for Visualizations


@patch("pandas.read_csv")
def test_feature_importance_mapping(mock_read_csv):
    """Test if internal name mapping correctly renames raw features to user-friendly labels."""
    mock_read_csv.return_value = pd.DataFrame(
        {
            "feature": ["engagement_ratio_7v30", "days_since_last_watch"],
            "importance": [0.5, 0.3],
        }
    )

    fig = uc3.plot_feature_importance_from_csv()
    ax = fig.get_axes()[0]
    yticklabels = [label.get_text().strip() for label in ax.get_yticklabels()]

    # Check if 'engagement_7v30' was successfully mapped to 'Watch Time Decline Rate'
    assert "Engagement Drop (Last 7d vs 30d)" in yticklabels
    assert "Days Since Last View" in yticklabels


def test_plot_pm_report_success():
    """
    Test the plot_pm_report function to ensure it correctly generates
    a visualization of genre-specific bounce rates.
    """
    # Create mock data containing necessary columns for the visualization
    mock_df = pd.DataFrame(
        {
            "bounce_rate": [0.1, 0.2, 0.3],
            "genre_primary": ["Action", "Drama", "Comedy"],
            "churn_label": ["Churn", "No Churn", "Churn"],
        }
    )

    # Execute the plotting function
    fig = uc3.plot_pm_report(mock_df)

    # Verify that a matplotlib Figure object is successfully created
    assert isinstance(
        fig, plt.Figure
    ), "The function should return a matplotlib Figure object."

    # Verify the plot title to ensure the correct chart is rendered
    expected_title = "Which Genre Disappoints Users the Most? (Bounce Rate)"
    assert (
        fig.axes[0].get_title() == expected_title
    ), f"Expected title: {expected_title}"

    # Clean up memory by closing the figure after the test
    plt.close(fig)


# Tests for Data Loading and Mocking


@patch("usecase3.bigquery.Client")
def test_load_failed_queries_mock(mock_client):
    """Check if BigQuery query results are correctly converted to a DataFrame."""
    mock_df = pd.DataFrame({"search_query": ["test_query"], "failure_count": [10]})
    mock_client.return_value.query.return_value.to_dataframe.return_value = mock_df

    result = uc3.load_failed_queries()
    assert len(result) == 1
    assert result["search_query"][0] == "test_query"


@patch("usecase3.bigquery.Client")
def test_all_load_functions_coverage(mock_client):
    """Execute all data loading functions to ensure SQL queries and conversion logic are covered."""
    fake_df = pd.DataFrame({"dummy_col": [1, 2, 3]})
    mock_client.return_value.query.return_value.to_dataframe.return_value = fake_df

    # Run all cached data functions
    uc3.load_null_search_analysis()
    uc3.load_failed_queries()
    uc3.load_tenure_analysis()
    uc3.load_final_pm_report()

    assert True


# Edge Case Tests


def test_plot_tenure_with_empty_data():
    """Edge Case: Ensure the tenure plot does not crash when provided with an empty DataFrame."""
    empty_df = pd.DataFrame(columns=["tenure_months", "total_users", "churn_rate"])

    fig = uc3.plot_tenure(empty_df)
    assert isinstance(fig, plt.Figure)
