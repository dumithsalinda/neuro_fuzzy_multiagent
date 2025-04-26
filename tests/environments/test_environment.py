import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from neuro_fuzzy_multiagent.env.abstraction import SimpleEnvironment
from neuro_fuzzy_multiagent.environment.transfer_learning import FeatureExtractor, transfer_learning
from neuro_fuzzy_multiagent.core.neural_networks.neural_network import FeedforwardNeuralNetwork


def test_simple_environment():
    env = SimpleEnvironment(dim=4)
    state = env.reset()
    assert state.shape == (4,)
    obs = env.perceive()
    assert np.allclose(state, obs)
    features = env.extract_features()
    assert np.allclose(state, features)


def test_feature_extractor():
    feat = FeatureExtractor(input_dim=4, output_dim=2)
    x = np.random.randn(4)
    fx = feat.extract(x)
    assert fx.shape == (2,)


def test_transfer_learning_workflow():
    env1 = SimpleEnvironment(dim=4)
    env2 = SimpleEnvironment(dim=4)
    feat = FeatureExtractor(4, 2)
    model = FeedforwardNeuralNetwork(input_dim=2, hidden_dim=3, output_dim=1)
    trained_model = transfer_learning(env1, env2, model, feat, steps=2)
    assert hasattr(trained_model, "forward")
