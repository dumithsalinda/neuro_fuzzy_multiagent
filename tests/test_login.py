import pytest
from unittest.mock import patch
import streamlit as st
from dashboard.login import login_form


def test_login_form_success(monkeypatch):
    st.session_state.clear()
    st.session_state["authenticated"] = False
    # Patch Streamlit widgets for username, password, and button
    with patch("dashboard.login.st", st), patch(
        "dashboard.login.st.text_input", side_effect=["admin", "password"]
    ), patch("dashboard.login.st.form_submit_button", return_value=True), patch(
        "dashboard.login.st.success"
    ), patch(
        "dashboard.login.st.rerun"
    ), patch(
        "dashboard.login.st.error"
    ) as mock_error:
        login_form()
        assert st.session_state["authenticated"] is True
        # Try wrong password
        st.session_state["authenticated"] = False
        with patch("dashboard.login.st.text_input", side_effect=["admin", "wrong"]):
            login_form()
            mock_error.assert_called()
