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
        raise ValueError(
            f"Unknown network type: {name}. Available: {list(NN_REGISTRY)}"
        )
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


# ---- Convolutional Neural Network (CNN) ----
@register_neural_network
class ConvolutionalNeuralNetwork(BaseNeuralNetwork):
    """
    Simple 1D CNN for plug-and-play extension.
    """
    def __init__(self, input_shape, num_filters, kernel_size, output_dim, activation=np.tanh):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.activation = activation
        # Initialize weights for 1D conv layer and FC output
        self.W_conv = np.random.randn(num_filters, input_shape, kernel_size) * 0.1
        self.b_conv = np.zeros((num_filters,))
        self.W_fc = np.random.randn(num_filters, output_dim) * 0.1
        self.b_fc = np.zeros(output_dim)

    def conv1d(self, x):
        # x: (batch, input_shape)
        batch, length = x.shape
        out_length = length - self.kernel_size + 1
        conv_out = np.zeros((batch, self.num_filters, out_length))
        for i in range(self.num_filters):
            for j in range(out_length):
                conv_out[:, i, j] = np.sum(x[:, j:j+self.kernel_size] * self.W_conv[i, :, :], axis=1) + self.b_conv[i]
        return self.activation(conv_out)

    def forward(self, x):
        # x: (batch, input_shape)
        conv_out = self.conv1d(x)  # (batch, num_filters, out_length)
        # Global average pooling
        pooled = np.mean(conv_out, axis=2)  # (batch, num_filters)
        out = np.dot(pooled, self.W_fc) + self.b_fc  # (batch, output_dim)
        return out

    def backward(self, x, y, lr=0.01):
        # Dummy implementation (not a real CNN backward)
        pass

    def evolutionary_update(self, mutation_rate=0.01):
        self.W_conv += np.random.randn(*self.W_conv.shape) * mutation_rate
        self.b_conv += np.random.randn(*self.b_conv.shape) * mutation_rate
        self.W_fc += np.random.randn(*self.W_fc.shape) * mutation_rate
        self.b_fc += np.random.randn(*self.b_fc.shape) * mutation_rate

# ---- RNN, LSTM, GRU, Autoencoder, RLAgent ----
@register_neural_network
class SimpleRNN(BaseNeuralNetwork):
    """
    Simple RNN for sequential data.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, activation=np.tanh):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.Wx = np.random.randn(input_dim, hidden_dim) * 0.1
        self.Wh = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.bh = np.zeros(hidden_dim)
        self.Wy = np.random.randn(hidden_dim, output_dim) * 0.1
        self.by = np.zeros(output_dim)

    def forward(self, x):
        # x: (seq_len, input_dim)
        h = np.zeros(self.hidden_dim)
        for t in range(x.shape[0]):
            h = self.activation(np.dot(x[t], self.Wx) + np.dot(h, self.Wh) + self.bh)
        out = np.dot(h, self.Wy) + self.by
        return out

    def backward(self, x, y, lr=0.01):
        pass

    def evolutionary_update(self, mutation_rate=0.01):
        self.Wx += np.random.randn(*self.Wx.shape) * mutation_rate
        self.Wh += np.random.randn(*self.Wh.shape) * mutation_rate
        self.bh += np.random.randn(*self.bh.shape) * mutation_rate
        self.Wy += np.random.randn(*self.Wy.shape) * mutation_rate
        self.by += np.random.randn(*self.by.shape) * mutation_rate

@register_neural_network
class SimpleLSTM(BaseNeuralNetwork):
    """
    Simple LSTM for sequential data (minimal, not optimized).
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # LSTM weights (minimal, not production-ready)
        self.Wf = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.1
        self.Wi = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.1
        self.Wo = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.1
        self.Wc = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.1
        self.bf = np.zeros(hidden_dim)
        self.bi = np.zeros(hidden_dim)
        self.bo = np.zeros(hidden_dim)
        self.bc = np.zeros(hidden_dim)
        self.Wy = np.random.randn(hidden_dim, output_dim) * 0.1
        self.by = np.zeros(output_dim)

    def forward(self, x):
        # x: (seq_len, input_dim)
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        for t in range(x.shape[0]):
            concat = np.concatenate([x[t], h])
            f = self._sigmoid(np.dot(concat, self.Wf) + self.bf)
            i = self._sigmoid(np.dot(concat, self.Wi) + self.bi)
            o = self._sigmoid(np.dot(concat, self.Wo) + self.bo)
            c_tilde = np.tanh(np.dot(concat, self.Wc) + self.bc)
            c = f * c + i * c_tilde
            h = o * np.tanh(c)
        out = np.dot(h, self.Wy) + self.by
        return out

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x, y, lr=0.01):
        pass

    def evolutionary_update(self, mutation_rate=0.01):
        for w in [self.Wf, self.Wi, self.Wo, self.Wc, self.Wy]:
            w += np.random.randn(*w.shape) * mutation_rate
        for b in [self.bf, self.bi, self.bo, self.bc, self.by]:
            b += np.random.randn(*b.shape) * mutation_rate

@register_neural_network
class SimpleGRU(BaseNeuralNetwork):
    """
    Simple GRU for sequential data (minimal, not optimized).
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # GRU weights
        self.Wz = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.1
        self.Wr = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.1
        self.Wh = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.1
        self.bz = np.zeros(hidden_dim)
        self.br = np.zeros(hidden_dim)
        self.bh = np.zeros(hidden_dim)
        self.Wy = np.random.randn(hidden_dim, output_dim) * 0.1
        self.by = np.zeros(output_dim)

    def forward(self, x):
        # x: (seq_len, input_dim)
        h = np.zeros(self.hidden_dim)
        for t in range(x.shape[0]):
            concat = np.concatenate([x[t], h])
            z = self._sigmoid(np.dot(concat, self.Wz) + self.bz)
            r = self._sigmoid(np.dot(concat, self.Wr) + self.br)
            concat_r = np.concatenate([x[t], r * h])
            h_hat = np.tanh(np.dot(concat_r, self.Wh) + self.bh)
            h = (1 - z) * h + z * h_hat
        out = np.dot(h, self.Wy) + self.by
        return out

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x, y, lr=0.01):
        pass

    def evolutionary_update(self, mutation_rate=0.01):
        for w in [self.Wz, self.Wr, self.Wh, self.Wy]:
            w += np.random.randn(*w.shape) * mutation_rate
        for b in [self.bz, self.br, self.bh, self.by]:
            b += np.random.randn(*b.shape) * mutation_rate

@register_neural_network
class SimpleAutoencoder(BaseNeuralNetwork):
    """
    Simple autoencoder for dimensionality reduction/anomaly detection.
    """
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_enc = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b_enc = np.zeros(hidden_dim)
        self.W_dec = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b_dec = np.zeros(input_dim)

    def forward(self, x):
        # x: (batch, input_dim)
        encoded = np.tanh(np.dot(x, self.W_enc) + self.b_enc)
        decoded = np.dot(encoded, self.W_dec) + self.b_dec
        return decoded

    def backward(self, x, y, lr=0.01):
        # Dummy implementation (not a real autoencoder backward)
        pass

    def evolutionary_update(self, mutation_rate=0.01):
        self.W_enc += np.random.randn(*self.W_enc.shape) * mutation_rate
        self.b_enc += np.random.randn(*self.b_enc.shape) * mutation_rate
        self.W_dec += np.random.randn(*self.W_dec.shape) * mutation_rate
        self.b_dec += np.random.randn(*self.b_dec.shape) * mutation_rate

@register_neural_network
class SimpleRLAgent(BaseNeuralNetwork):
    """
    Simple RL agent using Q-learning (table-based for illustration).
    """
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_table = np.zeros((state_dim, action_dim))
        self.last_state = None
        self.last_action = None

    def forward(self, state):
        # Returns action with highest Q-value
        return np.argmax(self.q_table[state])

    def backward(self, state, reward, next_state, lr=0.1, gamma=0.99):
        # Q-learning update
        a = self.forward(state)
        best_next = np.max(self.q_table[next_state])
        td_target = reward + gamma * best_next
        td_error = td_target - self.q_table[state, a]
        self.q_table[state, a] += lr * td_error

    def evolutionary_update(self, mutation_rate=0.01):
        self.q_table += np.random.randn(*self.q_table.shape) * mutation_rate


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
