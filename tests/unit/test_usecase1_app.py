"""
Tests for the Use Case 1 Streamlit UI (`usecase1_app.run_usecase_1`).

We exercise:
- Initial layout and key text (unit test).
- Interaction with the churn threshold slider (one-shot test).
- Edge case when no users match the filters.
"""

from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


def _build_app():
    """Wrapper so AppTest can execute the Use Case 1 page in isolation."""

    def _app():
        from usecase1 import (  # pylint: disable=import-outside-toplevel
            usecase1_app,
        )

        usecase1_app.run_usecase_1()

    return _app


@patch("usecase1.usecase1_app.st.plotly_chart")
@patch("usecase1.usecase1_app.uc1")
def test_initial_layout_and_text(mock_uc1, _mock_plot):
    """Unit test: title, question text, and step headers are rendered."""
    # Minimal fake data so the app can render
    fake_artifacts = {"config": {"N_CLUSTERS": 2}}
    fake_users = pd.DataFrame(
        {
            "churn_probability_pct": [90.0, 70.0],
            "segment": [0, 1],
            "subscription_plan": ["Basic", "Premium"],
        }
    )
    mock_uc1.load_artifacts.return_value = fake_artifacts
    mock_uc1.load_users_from_bq.return_value = fake_users
    mock_uc1.predict_churn_logic.return_value = fake_users
    mock_uc1.SUBSCRIPTION_PLANS = {"Basic": 9.99, "Premium": 19.99}
    mock_uc1.get_churn_dist_plot.return_value = object()
    mock_uc1.get_segment_pie_plot.return_value = object()

    app = _build_app()
    at = AppTest.from_function(app).run()

    # Title and Marcus' guiding question exist
    assert any("Churn Prediction" in t.value for t in at.title)
    assert any("Marcus' question" in m.value for m in at.markdown)

    # Step headers appear (STEP 1 and STEP 2 are required)
    assert any("STEP 1" in h.value for h in at.header)
    assert any("STEP 2" in h.value for h in at.header)


@patch("usecase1.usecase1_app.st.plotly_chart")
@patch("usecase1.usecase1_app.uc1")
def test_threshold_filter_changes_metrics(mock_uc1, _mock_plot):
    """One-shot test: moving the threshold filter reduces high-risk user count."""
    fake_artifacts = {"config": {"N_CLUSTERS": 1}}
    fake_users = pd.DataFrame(
        {
            "churn_probability_pct": [60.0, 80.0, 95.0],
            "segment": [0, 0, 0],
            "subscription_plan": ["Basic", "Basic", "Basic"],
        }
    )
    mock_uc1.load_artifacts.return_value = fake_artifacts
    mock_uc1.load_users_from_bq.return_value = fake_users
    mock_uc1.predict_churn_logic.return_value = fake_users
    mock_uc1.SUBSCRIPTION_PLANS = {"Basic": 9.99}
    mock_uc1.get_churn_dist_plot.return_value = object()
    mock_uc1.get_segment_pie_plot.return_value = object()

    app = _build_app()
    at = AppTest.from_function(app).run()

    # Default threshold = 85 → only users with prob >= 85 (one user: 95)
    high_risk_metric = next(m for m in at.metric if m.label == "High-Risk Users")
    assert high_risk_metric.value == "1"

    # Increase threshold to 98 → no users
    slider = next(s for s in at.slider if "Churn Threshold" in s.label)
    slider.set_value(98).run()
    high_risk_metric = next(m for m in at.metric if m.label == "High-Risk Users")
    assert high_risk_metric.value == "0"


@patch("usecase1.usecase1_app.st.plotly_chart")
@patch("usecase1.usecase1_app.uc1")
def test_no_users_matching_filters_edge_case(mock_uc1, _mock_plot):
    """Edge case: when filters exclude all users, a warning is shown."""
    fake_artifacts = {"config": {"N_CLUSTERS": 1}}
    fake_users = pd.DataFrame(
        {
            "churn_probability_pct": [10.0, 20.0],
            "segment": [0, 0],
            "subscription_plan": ["Basic", "Basic"],
        }
    )
    mock_uc1.load_artifacts.return_value = fake_artifacts
    mock_uc1.load_users_from_bq.return_value = fake_users
    mock_uc1.predict_churn_logic.return_value = fake_users
    mock_uc1.SUBSCRIPTION_PLANS = {"Basic": 9.99}
    mock_uc1.get_churn_dist_plot.return_value = object()
    mock_uc1.get_segment_pie_plot.return_value = object()

    app = _build_app()
    at = AppTest.from_function(app).run()

    # Set threshold very high so no users match
    slider = next(s for s in at.slider if "Churn Threshold" in s.label)
    slider.set_value(100).run()

    # Warning about no users should be present
    assert any(
        "No users match the current criteria" in w.body for w in at.warning
    )
