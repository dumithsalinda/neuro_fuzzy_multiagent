"""
test_neuro_fuzzy_agent.py

Tests for NeuroFuzzyAgent integration and action selection.
"""
import numpy as np
import pytest
from src.core.agents.agent import NeuroFuzzyAgent

def test_neuro_fuzzy_agent_action():
    nn_config = {"input_dim": 1, "output_dim": 2, "hidden_dim": 4}
    agent = NeuroFuzzyAgent(nn_config, None)
    obs = np.array([0.5])
    action = agent.act(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == (2,)
    # Check deterministic behavior for same observation if model is not stochastic
    action2 = agent.act(obs)
    assert np.allclose(action, action2)
