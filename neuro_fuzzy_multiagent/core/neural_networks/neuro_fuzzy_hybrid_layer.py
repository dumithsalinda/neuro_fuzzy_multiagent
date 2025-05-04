"""
neuro_fuzzy_hybrid_layer.py
General-purpose neuro-fuzzy hybrid layer for adaptive integration of neural and fuzzy outputs.
Goes beyond simple output fusion by learning to combine both sources adaptively.
"""
import numpy as np

class NeuroFuzzyHybridLayer:
    """
    Learns to combine neural and fuzzy outputs adaptively.
    By default, uses a trainable weighted sum (can be extended to a neural layer).
    """
    def __init__(self, output_dim, init_weight=0.5, learnable=True):
        self.output_dim = output_dim
        self.learnable = learnable
        # Initialize fusion weights (one per output dim)
        self.weight = np.ones(output_dim) * init_weight
        self.bias = np.zeros(output_dim)
        self.lr = 0.01  # Learning rate for weight updates

    def forward(self, neural_out, fuzzy_out):
        """
        Combines neural and fuzzy outputs using adaptive weights.
        Args:
            neural_out: np.ndarray, shape (..., output_dim)
            fuzzy_out: np.ndarray, shape (..., output_dim)
        Returns:
            hybrid_out: np.ndarray, shape (..., output_dim)
        """
        # Ensure correct shape
        neural_out = np.asarray(neural_out)
        fuzzy_out = np.asarray(fuzzy_out)
        # Weighted sum
        hybrid_out = self.weight * neural_out + (1 - self.weight) * fuzzy_out + self.bias
        return hybrid_out

    def backward(self, neural_out, fuzzy_out, target):
        """
        Simple gradient update for fusion weights (MSE loss).
        Args:
            neural_out, fuzzy_out: np.ndarray, (..., output_dim)
            target: np.ndarray, (..., output_dim)
        """
        if not self.learnable:
            return
        pred = self.forward(neural_out, fuzzy_out)
        grad = (pred - target) * (neural_out - fuzzy_out)  # dL/dw
        grad = np.mean(grad, axis=0)  # Average over batch
        self.weight -= self.lr * grad
        # Clamp weights to [0,1] for interpretability
        self.weight = np.clip(self.weight, 0, 1)

    def set_weights(self, weight, bias=None):
        self.weight = np.asarray(weight)
        if bias is not None:
            self.bias = np.asarray(bias)

# Example usage:
# layer = NeuroFuzzyHybridLayer(output_dim=2)
# hybrid_out = layer.forward(neural_out, fuzzy_out)
# layer.backward(neural_out, fuzzy_out, target)
