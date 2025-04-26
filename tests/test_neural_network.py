import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from src.core.neural_networks.neural_network import FeedforwardNeuralNetwork


def test_feedforward_neural_network_forward():
    nn = FeedforwardNeuralNetwork(input_dim=2, hidden_dim=3, output_dim=1)
    x = np.array([[0.5, -0.5]])  # batch size 1
    out = nn.forward(x)
    assert out.shape == (1, 1)
    # Output should be finite
    assert np.isfinite(out).all()


def test_feedforward_neural_network_init_params():
    nn = FeedforwardNeuralNetwork(2, 3, 1)
    assert nn.W1.shape == (2, 3)
    assert nn.b1.shape == (3,)
    assert nn.W2.shape == (3, 1)
    assert nn.b2.shape == (1,)
