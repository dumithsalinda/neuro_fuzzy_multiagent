import pytest
from unittest.mock import patch
import streamlit as st
from dashboard.intervention import render_intervention_log

def test_render_intervention_log_with_data():
    st.session_state.clear()
    st.session_state["interventions"] = [{"a": 1, "b": 2}]
    with patch("dashboard.intervention.st", st):
        render_intervention_log()

def test_render_intervention_log_no_data():
    st.session_state.clear()
    with patch("dashboard.intervention.st", st):
        render_intervention_log()
