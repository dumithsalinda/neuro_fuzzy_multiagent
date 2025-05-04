"""
rnn_agent.py
Agent that uses a SimpleRNN for sequential decision tasks.
"""
from neuro_fuzzy_multiagent.core.neural_networks.neural_network import SimpleRNN
from neuro_fuzzy_multiagent.core.plugins.registration_utils import register_plugin

@register_plugin("agent")
class RNNAgent:
    """
    Agent using a SimpleRNN model.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, policy=None):
        self.model = SimpleRNN(input_dim, hidden_dim, output_dim)
        self.policy = policy if policy is not None else self.default_policy

    def act(self, obs_seq):
        out = self.model.forward(obs_seq)
        return self.policy(out)

    def default_policy(self, out):
        # Greedy action selection
        return int(out.argmax())
