"""
neural_network.py
Minimal feedforward neural network class for NeuroFuzzyHybrid:
    Adaptive Neuro-Fuzzy Inference System (ANFIS)-like hybrid model.
    input_dim should match the feature vector dimension for the agent's input type (e.g., 768 for BERT, 512 for ResNet18).
    Combines neural network and fuzzy inference system for hybrid learning.
    Supports forward, backward, and evolutionary update stubs.
"""

import numpy as np

from abc import ABC, abstractmethod

# ---- Neural Network Registry ----
NN_REGISTRY = {}

def register_neural_network(cls):
    """
    Decorator to register a neural network class for plug-and-play discovery.
    Usage: @register_neural_network above your NN class.
    """
    NN_REGISTRY[cls.__name__] = cls
    return cls

def get_registered_networks():
    """
    Returns a dict of all registered neural network classes: {name: class}
    """
    return dict(NN_REGISTRY)

def create_network_by_name(name, *args, **kwargs):
    """
    Factory to instantiate a registered network by name.
    Example: create_network_by_name('FeedforwardNeuralNetwork', ...)
    """
    if name not in NN_REGISTRY:
        raise ValueError(f"Unknown network type: {name}. Available: {list(NN_REGISTRY)}")
    return NN_REGISTRY[name](*args, **kwargs)


class BaseNeuralNetwork(ABC):
    """
    Abstract base class for all neural network types.
    """
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x, y, lr=0.01):
        pass

    @abstractmethod
    def evolutionary_update(self, mutation_rate=0.01):
        pass

@register_neural_network
class FeedforwardNeuralNetwork(BaseNeuralNetwork):
    """
    Minimal feedforward neural network for hybrid neuro-fuzzy systems.
    input_dim should match the feature vector dimension for the agent's input type (e.g., 768 for BERT, 512 for ResNet18).
    Extendable for hybrid learning (backpropagation + evolution).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, activation="tanh"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        activation_map = {
            "tanh": np.tanh,
            "relu": lambda x: np.maximum(0, x),
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
            # Add more as needed
        }
        if callable(activation):
            self.activation = activation
        else:
            self.activation = activation_map.get(activation, np.tanh)
        # Weight initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)
        self.learning_rate = 0.01  # Default public learning rate

    def forward(self, x):
        """Forward pass."""
        h = self.activation(np.dot(x, self.W1) + self.b1)
        out = np.dot(h, self.W2) + self.b2
        return out

    def backward(self, x, y, lr=0.01):
        """Stub for backpropagation update. To be implemented."""
        pass

    def evolutionary_update(self, mutation_rate=0.01):
        """Stub for evolutionary update (mutation/crossover). To be implemented."""
        pass

# ---- Convolutional Neural Network (CNN) Template ----
@register_neural_network
class ConvolutionalNeuralNetwork(BaseNeuralNetwork):
    """
    Example CNN for plug-and-play extension. (This is a stub; actual implementation needed for real use.)
    """
    def __init__(self, input_shape, num_filters, kernel_size, output_dim, activation=np.tanh):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.activation = activation
        # Initialize CNN weights here (stub)

    def forward(self, x):
        # Implement actual CNN forward pass here
        raise NotImplementedError("ConvolutionalNeuralNetwork.forward() not implemented.")

    def backward(self, x, y, lr=0.01):
        # Implement CNN backpropagation here
        raise NotImplementedError("ConvolutionalNeuralNetwork.backward() not implemented.")

    def evolutionary_update(self, mutation_rate=0.01):
        # Implement evolutionary update for CNN
        raise NotImplementedError("ConvolutionalNeuralNetwork.evolutionary_update() not implemented.")

# ---- Developer Notes ----
"""
How to add a new neural network type:
1. Inherit from BaseNeuralNetwork.
2. Implement forward, backward, and evolutionary_update methods.
3. Use @register_neural_network decorator above your class.
4. Your network will be auto-discovered and available for agent/model configuration.

Example:
@register_neural_network
class MyCustomNetwork(BaseNeuralNetwork):
    ...

"""
