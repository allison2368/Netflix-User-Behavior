"""
Integration Tests for Use Case 3 Dashboard UI (usecase3_app.py)
Focuses on user interaction and session state navigation.
"""
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


@patch("usecase3_app.uc3.get_summary_metrics")
@patch("usecase3_app.uc3.plot_feature_importance_from_csv")
def test_app_initial_load(_mock_plot, mock_metrics):
    """Test if the app initializes correctly with the 'Main' view."""
    # Mocking return values to avoid BigQuery connection
    mock_metrics.return_value = (8.0, 15.0, 15.0)

    at = AppTest.from_file("../../usecase3_app.py").run()

    # Check initial session state
    assert at.session_state.detail_view == "Main"
    # Check if the header and subheader exist
    assert len(at.header) > 0
    assert at.subheader[1].value == "Main Causes of User Churn"


@patch("usecase3_app.uc3.get_summary_metrics")
@patch("usecase3_app.uc3.load_tenure_analysis")
def test_navigation_to_tenure_view(mock_tenure, mock_metrics):
    """Test navigation to Tenure view with correct data columns."""
    mock_metrics.return_value = (8.0, 15.0, 15.0)
    mock_tenure.return_value = pd.DataFrame(
        {
            "tenure_months": [1, 2, 3],
            "total_users": [100, 90, 80],
            "churn_rate": [5.0, 4.0, 3.0],
        }
    )

    at = AppTest.from_file("../../usecase3_app.py").run()

    # Button index 0: tenure analysis
    target_button = next(b for b in at.button if "Tenure" in b.label)
    target_button.click().run()

    # Verify behavior: session state should change to 'Tenure'
    assert at.session_state.detail_view == "Tenure"
    assert any("Subscription Tenure Analysis" in s.value for s in at.subheader)


@patch("usecase3_app.uc3.get_summary_metrics")
@patch("usecase3_app.uc3.load_final_pm_report")
def test_navigation_to_bounce_view(mock_bounce, mock_metrics):
    """Test navigation to Genre/Bounce view."""
    mock_metrics.return_value = (8.0, 15.0, 15.0)
    mock_bounce.return_value = pd.DataFrame(
        {
            "genre_primary": ["Action", "Comedy"],
            "churn_label": ["Churn", "Not Churn"],
            "bounce_rate": [0.2, 0.1],
        }
    )

    at = AppTest.from_file("../../usecase3_app.py").run()

    # Button index 1: genre bounce viz
    target_button = next(b for b in at.button if "Genre" in b.label)
    target_button.click().run()

    assert at.session_state.detail_view == "Genre"
    assert any("Where Users Lose Interest" in s.value for s in at.subheader)
    assert len(at.expander) > 0


@patch("usecase3_app.uc3.get_summary_metrics")
@patch("usecase3_app.uc3.load_failed_queries")
def test_navigation_to_search_view(mock_queries, mock_metrics):
    """Test navigation to Search view with matching columns for Seaborn."""
    mock_metrics.return_value = (8.0, 15.0, 15.0)
    mock_queries.return_value = pd.DataFrame(
        {"search_query": ["X-files"], "failure_count": [5]}
    )

    at = AppTest.from_file("../../usecase3_app.py").run()

    # Button index 2: search
    at.button[2].click().run()

    assert at.session_state.detail_view == "Search"
    assert len(at.table) > 0  # verify if the table for failed queries is rendered


@patch("usecase3_app.uc3.get_summary_metrics")
def test_back_to_overview_button(mock_metrics):
    """Test if the 'Back to Overview' button restores the 'Main' view."""
    mock_metrics.return_value = (8.0, 15.0, 15.0)

    at = AppTest.from_file("../../usecase3_app.py").run()

    # Navigate to Tenure first
    at.button[0].click().run()
    assert at.session_state.detail_view == "Tenure"

    # Click 'Back to Overview' button (this is the last button rendered in the Tenure view)
    at.button[-1].click().run()

    # Verify we are back to Main
    assert at.session_state.detail_view == "Main"
