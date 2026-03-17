"""
Tests for the landing page (`landing.show_landing_page`).

We use Streamlit's testing API to validate layout and interactions:
- Initial render shows the Netflix title and four profile buttons.
- Clicking a profile button updates `st.session_state.selected_user`.
- The hidden `landing-marker` div is present so CSS hooks work.
"""

from streamlit.testing.v1 import AppTest


def _build_app():
    """Small wrapper so AppTest can execute the landing page in isolation."""

    def _app():
        # Import inside the function so it is defined in the transient module
        # that AppTest executes.
        from landing import (  # pylint: disable=import-outside-toplevel
            show_landing_page,
        )

        show_landing_page()

    return _app


def test_initial_render_shows_title_and_four_profiles():
    """Unit test: landing page renders header and four profile buttons."""
    app = _build_app()
    at = AppTest.from_function(app).run()

    # One main title "Who's Watching?" rendered as markdown
    assert any("Who's Watching?" in block.value for block in at.markdown)

    # Four profile selection buttons (one per persona)
    assert len(at.button) == 4


def test_clicking_profile_sets_selected_user_and_reruns():
    """One-shot tests: clicking each profile selects the correct user."""
    app = _build_app()
    at = AppTest.from_function(app).run()

    # Map expected keys in the order they were declared in landing.py
    expected_keys = ["marcus", "sarah", "puja", "admin"]
    assert len(at.button) == 4

    for idx, expected in enumerate(expected_keys):
        # Fresh run each time so state doesn't leak between iterations
        at = AppTest.from_function(app).run()
        at.button[idx].click().run()
        assert at.session_state.selected_user == expected


def test_landing_marker_div_present():
    """Edge case: CSS marker div must exist for landing styles to apply."""
    app = _build_app()
    at = AppTest.from_function(app).run()

    marker_snippet = 'id="landing-marker"'
    assert any(marker_snippet in block.value for block in at.markdown)

