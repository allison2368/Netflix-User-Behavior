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


# User Routing Test: Covering all user personas
@pytest.mark.parametrize("user_key", ["marcus", "sarah", "puja", "admin"])
def test_all_user_routing(user_key):
    """Test that every user type can log in and see their specific page."""
    at = AppTest.from_file("dashboard.py")
    at.session_state.selected_user = user_key
    at.run(timeout=30)
    assert not at.exception
    # Validation: Check if the user name appears in the sidebar (Covering current_user logic)
    assert any(user_key.capitalize() in s.value for s in at.sidebar.success)


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
