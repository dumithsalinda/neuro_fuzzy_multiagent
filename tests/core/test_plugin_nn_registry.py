"""
Test plug-and-play neural network registry and config-driven instantiation.
"""

import numpy as np
import pytest

from neuro_fuzzy_multiagent.core.neural_networks.neural_network import (
    FeedforwardNeuralNetwork,
    create_network_by_name,
    get_registered_networks,
)


def test_nn_registry_discovery():
    registry = get_registered_networks()
    assert "FeedforwardNeuralNetwork" in registry
    # If CNN template is present, check that too
    assert "ConvolutionalNeuralNetwork" in registry


def test_create_feedforward_network():
    net = create_network_by_name(
        "FeedforwardNeuralNetwork",
        input_dim=3,
        hidden_dim=4,
        output_dim=2,
        activation=np.tanh,
    )
    x = np.ones(3)
    out = net.forward(x)
    assert out.shape == (2,)


def test_create_cnn_template():
    # Should raise NotImplementedError on forward
    net = create_network_by_name(
        "ConvolutionalNeuralNetwork",
        input_shape=(8, 8, 1),
        num_filters=4,
        kernel_size=3,
        output_dim=2,
        activation=np.tanh,
    )
    with pytest.raises(NotImplementedError):
        net.forward(np.ones((8, 8, 1)))


def test_invalid_nn_type():
    with pytest.raises(ValueError):
        create_network_by_name("NonexistentNetwork", foo=1)


def test_nn_config_loader(tmp_path):
    # Write a YAML config file
    yaml_content = """
    nn_config:
      nn_type: FeedforwardNeuralNetwork
      input_dim: 3
      hidden_dim: 4
      output_dim: 2
      activation: tanh
    """
    config_file = tmp_path / "nn_config.yaml"
    config_file.write_text(yaml_content)
    from neuro_fuzzy_multiagent.utils.config_loader import load_nn_config

    nn_config = load_nn_config(str(config_file))
    assert nn_config["nn_type"] == "FeedforwardNeuralNetwork"
    assert nn_config["input_dim"] == 3
    assert nn_config["hidden_dim"] == 4
    assert nn_config["output_dim"] == 2
