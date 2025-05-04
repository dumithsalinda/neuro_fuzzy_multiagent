"""
lstm_agent.py
Agent that uses a SimpleLSTM for sequential decision tasks.
"""
from neuro_fuzzy_multiagent.core.neural_networks.neural_network import SimpleLSTM
from neuro_fuzzy_multiagent.core.plugins.registration_utils import register_plugin

@register_plugin("agent")
class LSTMAgent:
    """
    Agent using a SimpleLSTM model.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, policy=None):
        self.model = SimpleLSTM(input_dim, hidden_dim, output_dim)
        self.policy = policy if policy is not None else self.default_policy

    def act(self, obs_seq):
        out = self.model.forward(obs_seq)
        return self.policy(out)

    def default_policy(self, out):
        return int(out.argmax())
