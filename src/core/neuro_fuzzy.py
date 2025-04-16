"""
neuro_fuzzy.py

Implements an ANFIS-like neuro-fuzzy hybrid model combining neural networks and fuzzy logic systems.
Supports both evolutionary and gradient-based learning for transfer learning, adaptation, and robust inference.

Classes:
- NeuroFuzzyHybrid: Integrates a FeedforwardNeuralNetwork with a FuzzyInferenceSystem for hybrid learning.

This module is central to experiments in adaptive, interpretable, and robust learning systems.
"""

import numpy as np
from .neural_network import FeedforwardNeuralNetwork
from .fuzzy_system import FuzzyInferenceSystem

class NeuroFuzzyHybrid:
    def explain_action(self, x):
        # Fuzzy rule activations
        rule_activations = []
        if hasattr(self.fis, "rules"):
            for rule in self.fis.rules:
                activation = 1.0
                for (i, fs) in rule.antecedents:
                    activation *= fs.membership(x[i])
                rule_activations.append(activation)
        # Neural net output
        nn_out = self.nn.forward(x)
        # Chosen action (argmax of neural net output)
        import numpy as np
        action = int(np.argmax(nn_out)) if hasattr(nn_out, "__len__") and len(nn_out) > 1 else float(nn_out)
        return {
            "rule_activations": rule_activations,
            "nn_output": nn_out.tolist() if hasattr(nn_out, "tolist") else nn_out,
            "chosen_action": action
        }

    """
    Adaptive Neuro-Fuzzy Inference System (ANFIS)-like hybrid model.
    nn_config['input_dim'] should match the feature vector dimension for the agent's input type (e.g., 768 for BERT, 512 for ResNet18).
    Combines neural network and fuzzy inference system for hybrid learning.

    Parameters
    ----------
    nn_config : dict
        Configuration for FeedforwardNeuralNetwork (e.g., layer sizes, activations).
    fis_config : dict, optional
        Configuration for FuzzyInferenceSystem (e.g., rule generation params, fuzzy sets).
        If provided, used for dynamic rule generation after initialization.
    """
    def __init__(self, nn_config, fis_config=None):
        self.nn = FeedforwardNeuralNetwork(**nn_config)
        self.fis = FuzzyInferenceSystem()
        # Optionally generate rules if info provided
        if fis_config is not None:
            # Example: {'X': ..., 'y': ..., 'fuzzy_sets_per_input': ...}
            if all(k in fis_config for k in ('X', 'y', 'fuzzy_sets_per_input')):
                self.fis.dynamic_rule_generation(
                    fis_config['X'], fis_config['y'], fis_config['fuzzy_sets_per_input']
                )

    def online_update(self, x, y, lr=0.01):
        """
        Online/continual learning update. x: input, y: target output.
        Calls neural network backward method if implemented.
        """
        if hasattr(self.nn, 'backward'):
            self.nn.backward(x, y, lr=lr)
        # Optionally update fuzzy rules in future

    def forward(self, x):
        """
        Forward pass through fuzzy system and neural network.

        Parameters
        ----------
        x : np.ndarray
            Input feature vector.
        Returns
        -------
        np.ndarray
            Output of the neural network after fuzzy inference.
        """
        fuzzy_out = self.fis.evaluate(x)
        # Ensure input to NN is 1D array
        if np.isscalar(fuzzy_out):
            fuzzy_out = np.array([fuzzy_out])
        nn_out = self.nn.forward(fuzzy_out)
        arr = np.asarray(nn_out)
        if arr.shape == (1, 1):
            return arr.flatten()
        if arr.shape == (1,):
            return arr
        if arr.ndim == 0:
            return arr.reshape(1)
        return arr


    def backward(self, x, y, lr=0.01):
        """
        Hybrid backpropagation update for the neural component.

        Parameters
        ----------
        x : np.ndarray
            Batch of input vectors.
        y : np.ndarray
            Batch of target outputs.
        lr : float, optional
            Learning rate for gradient descent.
        Updates only the neural network parameters (fuzzy system is static).
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

    def self_organize(self, mutation_rate=0.01, tune_fuzzy=True, rule_change=True):
        """
        Self-organization: adapts neural and fuzzy structure/parameters.
        - Mutates neural network weights.
        - Tunes fuzzy sets (random data for demonstration).
        - Adds/removes fuzzy rules (randomly, for demonstration).
        """
        # Mutate neural network weights
        self.evolutionary_update(mutation_rate)

        # Tune fuzzy sets (using random data for demonstration)
        if tune_fuzzy:
            for rule in self.fis.rules:
                for i, fs in rule.antecedents:
                    # Generate random data for tuning
                    data = np.random.randn(10) * 0.5 + 0.5  # Example: 10 samples
                    fs.tune(data)

        # Randomly add or remove a rule (demonstration only)
        if rule_change and hasattr(self.fis, 'rules') and len(self.fis.rules) > 0:
            import random
            if random.random() < 0.5 and len(self.fis.rules) > 1:
                # Remove a random rule
                idx = random.randint(0, len(self.fis.rules)-1)
                del self.fis.rules[idx]
            else:
                # Duplicate a random rule with small perturbation
                idx = random.randint(0, len(self.fis.rules)-1)
                rule = self.fis.rules[idx]
                new_conseq = rule.consequent + np.random.randn()*0.1
                new_conseq = np.clip(new_conseq, 0, 1)  # Clamp to [0, 1]
                new_rule = type(rule)(rule.antecedents, new_conseq)
                self.fis.add_rule(new_rule)

    def loss(self, x, y):
        """Mean squared error loss for current input x and target y."""
        pred = self.forward(x)
        return np.mean((pred - y) ** 2)
