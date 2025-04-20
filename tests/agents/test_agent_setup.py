import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from src.core.agents.agent import Agent

from dashboard.agent_setup import setup_agents_and_env

def test_setup_agents_and_env_initializes_state():
    st.session_state.clear()
    st.session_state["agent_type"] = "Tabular Q-Learning"
    st.session_state["agent_count"] = 2
    st.session_state["n_obstacles"] = 1
    with patch("dashboard.agent_setup.st", st), \
         patch("dashboard.agent_setup.initialize_env_and_agents", return_value=(MagicMock(), [MagicMock(), MagicMock()])):
        setup_agents_and_env()
        assert "env" in st.session_state
        assert "agents" in st.session_state
        assert "obs" in st.session_state
        assert st.session_state["step"] == 0
