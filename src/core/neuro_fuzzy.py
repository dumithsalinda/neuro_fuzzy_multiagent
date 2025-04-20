"""
neuro_fuzzy.py

Implements an ANFIS-like neuro-fuzzy hybrid model combining neural networks and fuzzy logic systems.
Supports both evolutionary and gradient-based learning for transfer learning, adaptation, and robust inference.

Classes:
- NeuroFuzzyHybrid: Integrates a FeedforwardNeuralNetwork with a FuzzyInferenceSystem for hybrid learning.

This module is central to experiments in adaptive, interpretable, and robust learning systems.
"""

import numpy as np
from src.core.neural_networks.neural_network import create_network_by_name, FeedforwardNeuralNetwork
from src.core.neural_networks.fuzzy_system import FuzzyInferenceSystem


class NeuroFuzzyHybrid:
    def update_nn_config(self, nn_config):
        """
        Update the neural network configuration at runtime.
        Supports switching network type (plug-and-play).
        """
        nn_type = nn_config.pop('nn_type', 'FeedforwardNeuralNetwork')
        self.nn = create_network_by_name(nn_type, **nn_config)

    def update_fis_config(self, fis_config):
        """
        Update the fuzzy inference system configuration at runtime.
        If dynamic rule generation keys are present, regenerate rules.
        """
        if fis_config is not None and all(
            k in fis_config for k in ("X", "y", "fuzzy_sets_per_input")
        ):
            self.fis.dynamic_rule_generation(
                fis_config["X"], fis_config["y"], fis_config["fuzzy_sets_per_input"]
            )

    def __init__(self, nn_config, fis_config=None):
        """
        nn_config example:
        nn_config:
            nn_type: FeedforwardNeuralNetwork  # or ConvolutionalNeuralNetwork, etc.
            input_dim: 4
            hidden_dim: 8
            output_dim: 2
            activation: np.tanh
        """
        nn_type = nn_config.pop('nn_type', 'FeedforwardNeuralNetwork')
        self.nn = create_network_by_name(nn_type, **nn_config)
        self.fis = FuzzyInferenceSystem()
        self.mode = "hybrid"  # 'neural', 'fuzzy', or 'hybrid'
        self.hybrid_weight = 0.5  # Default: equal weighting
        # Optionally generate rules if info provided
        if fis_config is not None:
            if all(k in fis_config for k in ("X", "y", "fuzzy_sets_per_input")):
                self.fis.dynamic_rule_generation(
                    fis_config["X"], fis_config["y"], fis_config["fuzzy_sets_per_input"]
                )

    def set_mode(self, mode, hybrid_weight=None):
        """
        Set the inference mode. mode: 'neural', 'fuzzy', or 'hybrid'.
        If 'hybrid', hybrid_weight sets the neural/fuzzy blend (0.0-1.0).
        """
        assert mode in ("neural", "fuzzy", "hybrid")
        self.mode = mode
        if hybrid_weight is not None:
            assert 0.0 <= hybrid_weight <= 1.0
            self.hybrid_weight = hybrid_weight

    def get_mode(self):
        """Return the current inference mode."""
        return self.mode

    def infer(self, x):
        """
        Inference according to the current mode.
        - 'neural': direct NN(x)
        - 'fuzzy': direct FIS(x)
        - 'hybrid': weighted sum of NN(x) and FIS(x)
        """
        # Defensive: ensure mode and hybrid_weight always present
        if not hasattr(self, "mode"):
            self.mode = "hybrid"
        if not hasattr(self, "hybrid_weight"):
            self.hybrid_weight = 0.5
        if self.mode == "neural":
            return self.nn.forward(x)
        elif self.mode == "fuzzy":
            return self.fis.evaluate(x)
        elif self.mode == "hybrid":
            fuzzy_out = self.fis.evaluate(x)
            nn_out = self.nn.forward(x)
            # Ensure both are arrays for weighted sum
            fuzzy_arr = np.asarray(fuzzy_out)
            nn_arr = np.asarray(nn_out)
            # Broadcast/reshape if needed
            if fuzzy_arr.shape != nn_arr.shape:
                if fuzzy_arr.size == 1:
                    fuzzy_arr = np.full_like(nn_arr, fuzzy_arr)
                elif nn_arr.size == 1:
                    nn_arr = np.full_like(fuzzy_arr, nn_arr)
            hybrid_weight = self.hybrid_weight
            return hybrid_weight * nn_arr + (1 - hybrid_weight) * fuzzy_arr
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def explain_action(self, x):
        # Fuzzy rule activations
        rule_activations = []
        if hasattr(self.fis, "rules"):
            for rule in self.fis.rules:
                activation = 1.0
                for i, fs in rule.antecedents:
                    activation *= fs.membership(x[i])
                rule_activations.append(activation)
        # Neural net output
        nn_out = self.nn.forward(x)
        import numpy as np

        action = (
            int(np.argmax(nn_out))
            if hasattr(nn_out, "__len__") and len(nn_out) > 1
            else float(nn_out)
        )
        return {
            "rule_activations": rule_activations,
            "nn_output": nn_out.tolist() if hasattr(nn_out, "tolist") else nn_out,
            "chosen_action": action,
        }

    def forward(self, x):
        """
        Forward pass (default: uses current mode).
        """
        return self.infer(x)

    def add_rule(self, antecedents, consequent, as_core=False):
        """
        Add a fuzzy rule to the system at runtime.
        antecedents: list of (input_index, FuzzySet)
        consequent: output value or class label
        as_core: if True, add as core rule; otherwise, as dynamic rule
        """
        from src.core.neural_networks.fuzzy_system import FuzzyRule

        rule = FuzzyRule(antecedents, consequent)
        self.fis.add_rule(rule, as_core=as_core)

    def prune_rule(self, rule_idx, from_core=False):
        """
        Remove a fuzzy rule by index (from core or dynamic rules).
        """
        if from_core:
            if 0 <= rule_idx < len(self.fis.core_rules):
                del self.fis.core_rules[rule_idx]
        else:
            if 0 <= rule_idx < len(self.fis.dynamic_rules):
                del self.fis.dynamic_rules[rule_idx]
        self.fis.rules = self.fis.core_rules + self.fis.dynamic_rules

    def list_rules(self):
        """
        Return a summary of all fuzzy rules (core + dynamic).
        """
        summaries = []
        for i, rule in enumerate(self.fis.core_rules):
            summaries.append(
                {
                    "type": "core",
                    "index": i,
                    "antecedents": rule.antecedents,
                    "consequent": rule.consequent,
                }
            )
        for i, rule in enumerate(self.fis.dynamic_rules):
            summaries.append(
                {
                    "type": "dynamic",
                    "index": i,
                    "antecedents": rule.antecedents,
                    "consequent": rule.consequent,
                }
            )
        return summaries

    def evolve_rules(self, recent_inputs=None, min_avg_activation=0.01):
        """
        Prune dynamic fuzzy rules with average firing strength below threshold over recent_inputs.
        Optionally, can add/tune rules in future.
        """
        if recent_inputs is None or len(self.fis.dynamic_rules) == 0:
            return False
        avg_activations = []
        for rule in self.fis.dynamic_rules:
            activations = []
            for x in recent_inputs:
                activation = 1.0
                for i, fs in rule.antecedents:
                    activation *= fs.membership(x[i])
                activations.append(activation)
            avg_activations.append(np.mean(activations))
        # Prune rules below threshold
        to_prune = [
            i for i, avg in enumerate(avg_activations) if avg < min_avg_activation
        ]
        # Prune in reverse order to keep indices valid
        for idx in sorted(to_prune, reverse=True):
            del self.fis.dynamic_rules[idx]
        self.fis.rules = self.fis.core_rules + self.fis.dynamic_rules
        return len(to_prune) > 0

    def auto_switch_mode(self, error_history, thresholds=None):
        """
        Adaptively switch mode based on recent error history.
        thresholds: dict with keys 'neural', 'fuzzy', 'hybrid' and float values.
        If error is high in current mode, switch to another.
        """
        if thresholds is None:
            thresholds = {"neural": 0.2, "fuzzy": 0.2, "hybrid": 0.15}
        if not error_history or len(error_history) < 3:
            return self.mode
        avg_error = np.mean(error_history[-5:])
        # Simple logic: if error too high in current mode, switch
        if self.mode == "neural" and avg_error > thresholds["neural"]:
            self.set_mode("hybrid")
        elif self.mode == "hybrid" and avg_error > thresholds["hybrid"]:
            self.set_mode("fuzzy")
        elif self.mode == "fuzzy" and avg_error > thresholds["fuzzy"]:
            self.set_mode("neural")
        return self.mode

    def set_learning_rate(self, lr):
        """
        Set learning rate for the neural network and (if supported) the FIS.
        """
        if hasattr(self.nn, "learning_rate"):
            self.nn.learning_rate = lr
        if hasattr(self.fis, "learning_rate"):
            self.fis.learning_rate = lr

    def get_learning_rate(self):
        """
        Get learning rate from the neural network (and/or FIS).
        """
        if hasattr(self.nn, "learning_rate"):
            return self.nn.learning_rate
        elif hasattr(self.fis, "learning_rate"):
            return self.fis.learning_rate
        else:
            return None

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
            if all(k in fis_config for k in ("X", "y", "fuzzy_sets_per_input")):
                self.fis.dynamic_rule_generation(
                    fis_config["X"], fis_config["y"], fis_config["fuzzy_sets_per_input"]
                )

    def online_update(self, x, y, lr=0.01):
        """
        Online/continual learning update. x: input, y: target output.
        Calls neural network backward method if implemented.
        """
        if hasattr(self.nn, "backward"):
            self.nn.backward(x, y, lr=lr)
        # Optionally update fuzzy rules in future

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
        dh = np.dot(error, self.nn.W2.T) * (1 - h**2)
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
        if rule_change and hasattr(self.fis, "rules") and len(self.fis.rules) > 0:
            import random

            if random.random() < 0.5 and len(self.fis.rules) > 1:
                # Remove a random rule
                idx = random.randint(0, len(self.fis.rules) - 1)
                del self.fis.rules[idx]
            else:
                # Duplicate a random rule with small perturbation
                idx = random.randint(0, len(self.fis.rules) - 1)
                rule = self.fis.rules[idx]
                new_conseq = rule.consequent + np.random.randn() * 0.1
                new_conseq = np.clip(new_conseq, 0, 1)  # Clamp to [0, 1]
                new_rule = type(rule)(rule.antecedents, new_conseq)
                self.fis.add_rule(new_rule)

    def loss(self, x, y):
        """Mean squared error loss for current input x and target y."""
        pred = self.forward(x)
        return np.mean((pred - y) ** 2)
