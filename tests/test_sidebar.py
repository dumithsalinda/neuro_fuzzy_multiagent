import pytest
from unittest.mock import patch
import streamlit as st
from dashboard.sidebar import render_sidebar

def test_render_sidebar_sets_session_state():
    st.session_state.clear()
    # Patch streamlit sidebar widgets to simulate user input
    with patch("dashboard.sidebar.st", st), \
         patch("dashboard.sidebar.st.sidebar.selectbox", return_value="Tabular Q-Learning"), \
         patch("dashboard.sidebar.st.sidebar.slider", side_effect=[3, 2]):
        render_sidebar()
    assert st.session_state["agent_type"] == "Tabular Q-Learning"
    assert st.session_state["agent_count"] == 3
    assert st.session_state["n_obstacles"] == 2
