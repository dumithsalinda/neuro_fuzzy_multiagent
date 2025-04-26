import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)
from neuro_fuzzy_multiagent.core.agents.meta_agent import MetaAgent


class DummyAgent:
    def __init__(self, value=0):
        self.value = value
        self._actions = []

    def act(self, obs, state=None):
        self._actions.append(obs)
        return self.value

    def observe(self, reward, next_state, done):
        pass


def test_meta_agent_switching():
    # Create two dummy agents: one always returns 0, one always returns 1
    agents = [(DummyAgent, {"value": 0}), (DummyAgent, {"value": 1})]

    # Selection strategy: pick agent with highest mean reward
    def select_best(perfs):
        return int(np.argmax(perfs))

    meta = MetaAgent(agents, selection_strategy=select_best, window=3)
    obs = 0
    # Simulate 3 steps for agent 0 (reward=0)
    for _ in range(3):
        a = meta.act(obs)
        meta.observe(0, None, False)
    # Simulate 3 steps for agent 1 (reward=1)
    for _ in range(3):
        a = meta.act(obs)
        meta.observe(1, None, False)
    # After exploration, agent 1 should be selected
    assert meta.active_idx == 1
    # Now, if agent 0 gets better rewards, meta-agent should switch
    for _ in range(3):
        meta.active_idx = 0
        meta.active_agent = meta.candidate_agents[0]
        meta.observe(2, None, False)
    meta.maybe_switch_agent()
    assert meta.active_idx == 0
