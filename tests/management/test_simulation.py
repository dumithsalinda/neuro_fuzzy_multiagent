import pytest
import streamlit as st
from dashboard.simulation import simulate_step, run_batch_experiments, som_group_agents
from unittest.mock import patch, MagicMock
import numpy as np


def setup_streamlit_session(agents=2, obs_dim=3):
    st.session_state.clear()
    # Patch agents so that act returns 0 and observe does nothing
    agent1 = MagicMock()
    agent2 = MagicMock()
    agent1.group = 0
    agent2.group = 1
    agent1.last_rule = None
    agent2.last_rule = None
    # Patch __class__ name to not be DQNAgent
    agent1.__class__ = type("TestAgent", (), {})
    agent2.__class__ = type("TestAgent", (), {})
    agent1.act.return_value = 0
    agent2.act.return_value = 0
    agent1.observe.return_value = None
    agent2.observe.return_value = None
    st.session_state["agents"] = [agent1, agent2]
    st.session_state["multiagent_system"] = MagicMock()
    st.session_state["obs"] = np.zeros((agents, obs_dim)).tolist()
    st.session_state["rewards"] = [1.0, 2.0]
    st.session_state["step"] = 0
    st.session_state["env"] = MagicMock()
    st.session_state["feedback"] = {}
    st.session_state["online_learning_enabled"] = True
    st.session_state["adversarial_enabled"] = False
    st.session_state["adversarial_agents"] = []
    st.session_state["adversarial_type"] = "None"
    st.session_state["adversarial_strength"] = 0.1


def test_run_batch_experiments_output():
    results = run_batch_experiments(1, [2], [1], 2, fast_mode=True)
    assert isinstance(results, list)
    assert len(results) == 1
    assert "mean_reward" in results[0]


def test_som_group_agents_handles_missing_agents():
    st.session_state.clear()
    with patch("dashboard.simulation.st", st):
        som_group_agents()  # Should warn and return without error
