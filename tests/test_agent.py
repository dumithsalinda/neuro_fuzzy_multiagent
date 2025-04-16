"""
test_agent.py

Tests for the Agent class in agent.py.
"""

import numpy as np
import pytest
from src.core.agent import Agent

class DummyModel:
    def forward(self, x):
        return x * 2

def test_agent_act_and_observe():
    model = DummyModel()
    agent = Agent(model)
    obs = np.ones(3)
    action = agent.act(obs)
    assert action.shape == obs.shape
    agent.observe(1.5)
    assert agent.total_reward == 1.5
    agent.observe(-0.5)
    assert agent.total_reward == 1.0

def test_agent_reset():
    model = DummyModel()
    agent = Agent(model)
    obs = np.ones(2)
    agent.act(obs)
    agent.observe(2.0)
    agent.reset()
    assert agent.last_action is None
    assert agent.last_observation is None
    assert agent.total_reward == 0

def test_agent_custom_policy():
    def greedy_policy(obs, model):
        return model.forward(obs)
    model = DummyModel()
    agent = Agent(model, policy=greedy_policy)
    obs = np.ones(4)
    action = agent.act(obs)
    np.testing.assert_array_equal(action, obs * 2)
