"""
autoencoder_agent.py
Agent that uses a SimpleAutoencoder for anomaly detection or compression.
"""
from neuro_fuzzy_multiagent.core.neural_networks.neural_network import SimpleAutoencoder
from neuro_fuzzy_multiagent.core.plugins.registration_utils import register_plugin
import numpy as np

@register_plugin("agent")
class AutoencoderAgent:
    """
    Agent using a SimpleAutoencoder model.
    """
    def __init__(self, input_dim, hidden_dim, threshold=1.0):
        self.model = SimpleAutoencoder(input_dim, hidden_dim)
        self.threshold = threshold

    def is_anomaly(self, obs):
        recon = self.model.forward(obs)
        error = np.mean((obs - recon) ** 2)
        return error > self.threshold

    def compress(self, obs):
        encoded = np.tanh(np.dot(obs, self.model.W_enc) + self.model.b_enc)
        return encoded
