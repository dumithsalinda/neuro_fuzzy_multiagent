import numpy as np
import pytest

from neuro_fuzzy_multiagent.core.agents.dqn_agent import DQNAgent
from neuro_fuzzy_multiagent.core.agents.meta_agent import MetaAgent
from neuro_fuzzy_multiagent.core.agents.tabular_q_agent import TabularQLearningAgent


class DummyEnv:
    def reset(self):
        return 0

    def step(self, action):
        # Reward is higher for action 1
        return 0, 1 if action == 1 else 0, True, {}


def test_meta_agent_switching():
    # Two candidate agents: one always returns 0, one always returns 1
    class Always0Agent:
        def act(self, obs, state=None):
            return 0

        def observe(self, r, n, d):
            pass

        def reset(self):
            pass

    class Always1Agent:
        def act(self, obs, state=None):
            return 1

        def observe(self, r, n, d):
            pass

        def reset(self):
            pass

    meta = MetaAgent(
        candidate_agents=[(Always0Agent, {}), (Always1Agent, {})], window=3
    )
    env = DummyEnv()
    obs = env.reset()
    # Exploration phase: try each agent for 'window' steps
    for _ in range(meta.window * 2):
        action = meta.act(obs)
        _, reward, done, _ = env.step(action)
        meta.observe(reward, obs, done)
    # After exploration, should switch to Always1Agent
    assert meta.active_idx == 1


def test_meta_agent_reset():
    meta = MetaAgent(
        candidate_agents=[
            (TabularQLearningAgent, {"n_states": 2, "n_actions": 2}),
            (DQNAgent, {"state_dim": 2, "action_dim": 2}),
        ]
    )
    meta.reset()
    assert meta.active_idx == 0
    assert all(len(h) == 0 for h in meta.perf_history)
