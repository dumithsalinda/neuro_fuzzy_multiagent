import numpy as np

from neuro_fuzzy_multiagent.core.agents.meta_rl_agent import MetaRLAgent


class DummyBaseAgent:
    def __init__(self, lr=0.1):
        self.lr = lr
        self.actions = []
        self.obs = []

    def act(self, obs, state=None):
        return 0

    def observe(self, reward, next_state, done):
        self.obs.append((reward, next_state, done))

    def reset(self):
        self.obs = []
        self.lr = 0.1


def test_meta_rl_agent_meta_update():
    agent = MetaRLAgent(DummyBaseAgent, {"lr": 0.1}, meta_lr=0.05, meta_window=5)
    # Simulate improving rewards
    for _ in range(5):
        agent.observe(1.0, None, False)
    assert agent.meta_state["lr"] > 0.1
    # Simulate decreasing rewards
    for _ in range(5):
        agent.observe(-1.0, None, False)
    assert agent.meta_state["lr"] < 0.15
    # Check base agent's learning rate is updated
    assert abs(agent.base_agent.lr - agent.meta_state["lr"]) < 1e-6
