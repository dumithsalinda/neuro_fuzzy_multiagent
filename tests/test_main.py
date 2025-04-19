import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from dashboard.main import main

def test_main_unauthenticated_shows_login():
    st.session_state.clear()
    st.session_state["authenticated"] = False
    with patch("dashboard.main.login_form") as mock_login_form:
        main()
        mock_login_form.assert_called()

def test_main_authenticated_runs_dashboard():
    st.session_state.clear()
    st.session_state["authenticated"] = True
    with patch("dashboard.main.st", st), \
         patch("dashboard.main.collaborative_experiments_panel"), \
         patch("dashboard.main.simulate_step"), \
         patch("dashboard.main.som_group_agents"), \
         patch("dashboard.main.run_batch_experiments", return_value=[]), \
         patch("dashboard.main.render_agent_positions"), \
         patch("dashboard.main.render_group_modules"), \
         patch("dashboard.main.render_group_knowledge"):
        main()
