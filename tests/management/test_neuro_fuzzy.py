import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from neuro_fuzzy_multiagent.core.neural_networks.fuzzy_system import (
    FuzzyInferenceSystem,
    FuzzySet,
)
from neuro_fuzzy_multiagent.core.neural_networks.neural_network import (
    FeedforwardNeuralNetwork,
)
from neuro_fuzzy_multiagent.core.neuro_fuzzy import NeuroFuzzyHybrid


def test_neuro_fuzzy_hybrid_forward():
    # Setup fuzzy system
    fs_low = FuzzySet("low", [0.0, 1.0])
    fs_high = FuzzySet("high", [5.0, 1.0])
    fuzzy_sets = [[fs_low, fs_high]]
    X = np.array([[0.0], [5.0]])
    y = np.array([0.0, 1.0])
    fis_config = {}
    fis = FuzzyInferenceSystem()
    fis.dynamic_rule_generation(X, y, fuzzy_sets)
    # Setup neural network: 1 input (from FIS), 2 hidden, 1 output
    nn_config = dict(input_dim=1, hidden_dim=2, output_dim=1)
    nf = NeuroFuzzyHybrid(nn_config, fis_config={})
    nf.fis = fis  # inject configured FIS
    # Forward pass: input should propagate through FIS to NN
    out0 = nf.forward([0.0])
    out5 = nf.forward([5.0])
    # Accept both (1, 1) and (1,) or scalar
    assert np.shape(out0) in [(1, 1), (1,), ()]
    assert np.shape(out5) in [(1, 1), (1,), ()]
    # Output should be finite
    assert np.isfinite(out0).all() and np.isfinite(out5).all()


def test_neuro_fuzzy_hybrid_backward():
    # Setup fuzzy system
    fs_low = FuzzySet("low", [0.0, 1.0])
    fs_high = FuzzySet("high", [5.0, 1.0])
    fuzzy_sets = [[fs_low, fs_high]]
    X = np.array([[0.0], [5.0]])
    y = np.array([0.0, 1.0])
    fis = FuzzyInferenceSystem()
    fis.dynamic_rule_generation(X, y, fuzzy_sets)
    nn_config = dict(input_dim=1, hidden_dim=2, output_dim=1)
    nf = NeuroFuzzyHybrid(nn_config, fis_config={})
    nf.fis = fis
    # Use a batch of two diverse inputs for nonzero gradient
    batch_x = [list(X[0]), list(X[1])]
    batch_y = np.array([[0.5], [0.8]])
    W2_before = nf.nn.W2.copy()
    nf.backward(batch_x, batch_y, lr=0.01)
    assert not np.allclose(nf.nn.W2, W2_before)


def test_neuro_fuzzy_hybrid_evolutionary_update():
    nn_config = dict(input_dim=1, hidden_dim=2, output_dim=1)
    nf = NeuroFuzzyHybrid(nn_config, fis_config={})
    W1_before = nf.nn.W1.copy()
    nf.evolutionary_update(mutation_rate=0.1)
    assert not np.allclose(nf.nn.W1, W1_before)


def test_neuro_fuzzy_hybrid_loss():
    nn_config = dict(input_dim=1, hidden_dim=2, output_dim=1)
    nf = NeuroFuzzyHybrid(nn_config, fis_config={})
    x = [0.0]
    y = np.array([[0.0]])
    loss_val = nf.loss(x, y)
    assert isinstance(loss_val, float) or np.isscalar(loss_val)
