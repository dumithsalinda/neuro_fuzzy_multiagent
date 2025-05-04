"""
cnn_agent.py
Agent that uses a ConvolutionalNeuralNetwork for structured/spatial data.
"""
from neuro_fuzzy_multiagent.core.neural_networks.neural_network import ConvolutionalNeuralNetwork
from neuro_fuzzy_multiagent.core.plugins.registration_utils import register_plugin
import numpy as np

@register_plugin("agent")
class CNNAgent:
    """
    Agent using a ConvolutionalNeuralNetwork model.
    """
    def __init__(self, input_shape, num_filters, kernel_size, output_dim, activation=np.tanh, policy=None):
        self.model = ConvolutionalNeuralNetwork(input_shape, num_filters, kernel_size, output_dim, activation)
        self.policy = policy if policy is not None else self.default_policy

    def act(self, obs):
        out = self.model.forward(obs)
        return self.policy(out)

    def default_policy(self, out):
        return int(out.argmax())
