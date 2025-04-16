"""
neuro_fuzzy.py
ANFIS-like neuro-fuzzy hybrid model combining neural and fuzzy logic.
Supports both evolutionary and gradient-based learning.
"""

import numpy as np
from .neural_network import FeedforwardNeuralNetwork
from .fuzzy_system import FuzzyInferenceSystem

class NeuroFuzzyHybrid:
    """
    Adaptive Neuro-Fuzzy Inference System (ANFIS)-like hybrid model.
    Combines neural network and fuzzy inference system.
    Supports hybrid learning (evolution + backpropagation).
    """
    def __init__(self, nn_config, fis_config):
        self.nn = FeedforwardNeuralNetwork(**nn_config)
        self.fis = FuzzyInferenceSystem(**fis_config)

    def forward(self, x):
        """Forward pass through fuzzy system and neural network."""
        fuzzy_out = self.fis.evaluate(x)
        nn_out = self.nn.forward(fuzzy_out)
        return nn_out

    def backward(self, x, y, lr=0.01):
        """
        Hybrid backpropagation update:
        - For batch input: evaluate each x through fuzzy system, stack results
        - Forward pass: batch_fuzzy_out -> neural network
        - Compute MSE loss and gradients for neural network weights only
        """
        # Batch fuzzy evaluation
        batch_fuzzy_out = np.array([self.fis.evaluate(xi) for xi in x])
        if batch_fuzzy_out.ndim == 1:
            batch_fuzzy_out = batch_fuzzy_out.reshape(-1, 1)
        y = np.atleast_2d(y)
        pred = self.nn.forward(batch_fuzzy_out)
        error = pred - y
        h = self.nn.activation(np.dot(batch_fuzzy_out, self.nn.W1) + self.nn.b1)
        dW2 = np.dot(h.T, error)
        db2 = np.sum(error, axis=0)
        dh = np.dot(error, self.nn.W2.T) * (1 - h ** 2)
        dW1 = np.dot(batch_fuzzy_out.T, dh)
        db1 = np.sum(dh, axis=0)
        self.nn.W2 -= lr * dW2
        self.nn.b2 -= lr * db2
        self.nn.W1 -= lr * dW1
        self.nn.b1 -= lr * db1

    def evolutionary_update(self, mutation_rate=0.01):
        """
        Hybrid evolutionary update:
        - Mutate neural network weights with Gaussian noise
        - (Extendable to fuzzy rule/parameter mutation)
        """
        self.nn.W1 += np.random.randn(*self.nn.W1.shape) * mutation_rate
        self.nn.W2 += np.random.randn(*self.nn.W2.shape) * mutation_rate
        self.nn.b1 += np.random.randn(*self.nn.b1.shape) * mutation_rate
        self.nn.b2 += np.random.randn(*self.nn.b2.shape) * mutation_rate

    def loss(self, x, y):
        """Mean squared error loss for current input x and target y."""
        pred = self.forward(x)
        return np.mean((pred - y) ** 2)
