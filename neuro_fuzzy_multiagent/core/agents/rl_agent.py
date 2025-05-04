"""
rl_agent.py
Agent that uses a SimpleRLAgent for tabular Q-learning.
"""
from neuro_fuzzy_multiagent.core.neural_networks.neural_network import SimpleRLAgent
from neuro_fuzzy_multiagent.core.plugins.registration_utils import register_plugin

@register_plugin("agent")
class RLAgent:
    """
    Agent using a SimpleRLAgent (tabular Q-learning).
    """
    def __init__(self, state_dim, action_dim):
        self.model = SimpleRLAgent(state_dim, action_dim)

    def act(self, state):
        return self.model.forward(state)

    def learn(self, state, reward, next_state, lr=0.1, gamma=0.99):
        self.model.backward(state, reward, next_state, lr, gamma)
