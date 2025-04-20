import pytest
import streamlit as st
from dashboard.simulation import simulate_step, run_batch_experiments, som_group_agents
from unittest.mock import patch, MagicMock
import numpy as np

def setup_streamlit_session(agents=2, obs_dim=3):
    st.session_state.clear()
    st.session_state["agents"] = [MagicMock(group=0), MagicMock(group=1)]
    st.session_state["multiagent_system"] = MagicMock()
    st.session_state["obs"] = np.zeros((agents, obs_dim)).tolist()
    st.session_state["rewards"] = [1.0, 2.0]
    st.session_state["step"] = 0

def test_simulate_step_records_episode_memory():
    setup_streamlit_session()
    with patch("dashboard.simulation.st", st):
        simulate_step()
    assert "episode_memory" in st.session_state
    assert len(st.session_state["episode_memory"]) == 1
    assert isinstance(st.session_state["episode_memory"][0], list)

def test_run_batch_experiments_output():
    results = run_batch_experiments(1, [2], [1], 2, fast_mode=True)
    assert isinstance(results, list)
    assert len(results) == 1
    assert "mean_reward" in results[0]

def test_som_group_agents_handles_missing_agents():
    st.session_state.clear()
    with patch("dashboard.simulation.st", st):
        som_group_agents()  # Should warn and return without error
