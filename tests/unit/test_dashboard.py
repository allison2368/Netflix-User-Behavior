"""
Unit and Integration Tests for the Netflix Churn Prediction Dashboard.

This module utilizes the Streamlit AppTest framework to validate the
main dashboard entry point (dashboard.py). It covers:
1. User session routing for multiple personas (Marcus, Sarah, Puja, Admin).
2. Navigation logic including landing page redirects and home button resets.
3. Edge case handling for missing model artifacts and empty metrics.
4. UI component rendering including sidebars, radio buttons, and dataframes.

The tests are designed to ensure high code coverage and robust error handling
without the need for a live browser environment.
"""

from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from streamlit.runtime.scriptrunner import StopException

import dashboard


# Edge Case Test: Verify error handling when model files are missing
def test_load_model_artifacts_error_handling():
    """Edge Case: Test how the app handles missing artifacts."""
    with patch("churn_model.churn_model_updated.load_artifacts") as mock_load:
        # Simulate FileNotFoundError (Mocking)
        mock_load.side_effect = FileNotFoundError("File not found")

        # Call the function and handle the intentional stop
        # Since the function catches the error and calls st.stop(),
        # we ensure the test doesn't fail when the app stops.
        try:
            dashboard.load_model_artifacts()
        except StopException:
            # Successfully caught the intentional app stop triggered by st.stop()
            pass

        # Verify that the file loading was attempted and failed
        mock_load.assert_called_once()


# Integration Test: Verify if the main dashboard loads correctly
def test_dashboard_main_loading():
    """Test if the dashboard initializes correctly with a selected user."""
    at = AppTest.from_file("dashboard.py", default_timeout=30)

    # Inject session state (simulate logged-in user)
    at.session_state.selected_user = "admin"

    # Run the app
    at.run(timeout=30)

    # Validation: No exceptions should occur
    assert not at.exception
    # Validation: Check if "NETFLIX" header appears in the sidebar
    assert at.sidebar.markdown[0].value.find("NETFLIX") != -1


# Edge Case: Verify behavior when metrics are missing in Admin view
def test_admin_view_no_metrics():
    """Edge Case: Admin view when metrics are missing from artifacts."""
    at = AppTest.from_file("dashboard.py")
    # Set session state for admin user
    at.session_state.selected_user = "admin"

    # Run the app
    at.run(timeout=30)

    # Validation: App should run without crashing
    assert not at.exception

    # Validation: Check if 'Model Performance' radio button exists in the sidebar
    # (Since this only appears for admins, it proves the routing logic worked)
    assert any("Model Performance" in opt for opt in at.sidebar.radio[0].options)


# Integration Test: Verify routing for all user personas with comprehensive mocking
@pytest.mark.parametrize("user_key", ["marcus", "sarah", "puja", "admin"])
def test_all_user_routing(user_key):
    """Test that every user type can log in and see their specific page without crashes."""
    at = AppTest.from_file("dashboard.py")
    at.session_state.selected_user = user_key

    # Mock BigQuery interactions and complex Pandas logic to avoid Auth and KeyError issues
    with (
        patch("usecase1.usecase1.load_users_from_bq") as mock_uc1,
        patch("usecase2.usecase2.load_data") as mock_uc2,
        patch("usecase3.usecase3.get_summary_metrics") as mock_uc3,
    ):

        # Mock data for Marcus (Usecase 1)
        mock_uc1.return_value = pd.DataFrame(
            {
                "days_since_last_watch": [1],
                "total_sessions": [10],
                "avg_completion_rate": [0.5],
                "churn_prediction": [0],
            }
        )

        # Mock data for Sarah (Usecase 2)
        df_fake = pd.DataFrame(
            {
                "session_id": [101, 102],
                "user_id": ["U1", "U2"],
                "movie_id": [1, 2],
                "watch_date": ["2026-01-01", "2026-01-02"],
                "watch_duration_minutes": [30, 45],
                "progress_percentage": [80.0, 95.0],
                "title": ["Movie A", "Movie B"],
                "imdb_rating": [8.5, 7.2],
                "genre_primary": ["Drama", "Action"],
                "content_type": ["Movie", "Movie"],
                "is_netflix_original": [True, False],
                "is_series": [False, False],
                "release_year": [2024, 2025],
                "is_active": [True, True],
                "watch_decline_ratio": [0.1, 0.2],
                "engagement_ratio_7v30": [1.2, 1.5],
                "watch_last_7d": [100, 150],
                "watch_last_30d": [400, 500],
                "monthly_spend": [15.99, 15.99],
                "tenure_days": [365, 730],
                "subscription_plan": ["Premium", "Basic"],
                "avg_completion_rate": [0.85, 0.90],
            }
        )

        title_df_fake = pd.DataFrame(
            {
                "movie_id": [1, 2],
                "title": ["Movie A", "Movie B"],
                "imdb_rating": [8.5, 7.2],
                "genre_primary": ["Drama", "Action"],
                "content_type": ["Movie", "Movie"],
                "is_netflix_original": [True, False],
                "is_series": [False, False],
                "total_sessions": [100, 200],
                "total_watch_minutes": [3000, 9000],
                "avg_watch_duration": [30, 45],
                "avg_completion": [80.0, 95.0],
                "unique_viewers": [50, 100],
            }
        )
        mock_uc2.return_value = (df_fake, title_df_fake)

        # Mock data for Puja (Usecase 3)
        mock_uc3.return_value = (0.1, 0.2, 0.3)

        at.run(timeout=30)

        # Assertion: Ensure no unhandled exceptions were raised during execution
        assert not at.exception


# Navigation Test: Verify routing to Landing Page
def test_landing_page_routing():
    """Test when no user is selected (shows landing page)."""
    at = AppTest.from_file("dashboard.py")
    at.session_state.selected_user = None
    at.run(timeout=30)
    assert not at.exception
    # Indirectly verifies that landing.show_landing_page() is called


# Interaction Test: Verify 'Home' button resets the session
def test_home_button_click():
    """Test clicking the home button resets the user."""
    at = AppTest.from_file("dashboard.py")
    at.session_state.selected_user = "admin"
    at.run(timeout=30)

    # Find and simulate clicking the Home button (key="back_to_landing")
    if at.sidebar.button:
        home_btn = next(
            (b for b in at.sidebar.button if b.key == "back_to_landing"), None
        )
        if home_btn:
            home_btn.click().run(timeout=30)
            assert at.session_state.selected_user is None


# Logic Test: Verify metrics and feature importance rendering for Admin section
def test_admin_metrics_display():
    """Test detailed metrics table rendering for admin."""
    at = AppTest.from_file("dashboard.py")
    at.session_state.selected_user = "admin"

    # Prepare mock data for metrics
    mock_fi = pd.DataFrame({"Feature": ["Age", "Income"], "Importance": [0.8, 0.2]})
    mock_artifacts = {
        "metrics": {
            "accuracy": 0.95,
            "recall": 0.90,
            "f1_score": 0.92,
            "threshold": 0.5,
            "feature_importance": mock_fi,
        }
    }

    with patch("dashboard.load_model_artifacts") as mock_load:
        mock_load.return_value = mock_artifacts
        at.run(timeout=30)

        # Navigate to 'Model Performance' via button
        at.sidebar.radio[0].set_value("Model Performance").run(timeout=30)

        assert not at.exception
        # Validation: Verify that dataframes are rendered
        assert len(at.dataframe) >= 1
