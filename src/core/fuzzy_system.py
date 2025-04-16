"""
fuzzy_system.py
Fuzzy set, fuzzy rule, and fuzzy inference system classes.
Supports dynamic rule generation and self-tuning membership functions.
"""

import numpy as np

class FuzzySet:
    """
    Represents a fuzzy set with a membership function.
    Supports self-tuning of membership parameters.
    """
    def __init__(self, name, params):
        self.name = name
        self.params = params  # e.g., [center, width]

    def membership(self, x):
        """Gaussian membership function (example)."""
        c, w = self.params
        return np.exp(-0.5 * ((x - c) / w) ** 2)

    def tune(self, data):
        """Self-tune the center and width using data (simple mean/std update)."""
        # data: list or array of samples belonging to this fuzzy set
        if len(data) > 0:
            c = np.mean(data)
            w = np.std(data) + 1e-6  # avoid zero std
            self.params = [c, w]

class FuzzyRule:
    """
    Represents a fuzzy rule (IF-THEN) with antecedents and consequents.
    antecedents: list of (input_index, FuzzySet)
    consequent: output value (float or class)
    """
    def __init__(self, antecedents, consequent):
        self.antecedents = antecedents  # list of (input_index, FuzzySet)
        self.consequent = consequent    # output value

class FuzzyInferenceSystem:
    """
    Fuzzy inference system with dynamic rule generation and evaluation.
    """
    def __init__(self):
        self.rules = []

    def add_rule(self, rule):
        self.rules.append(rule)

    def dynamic_rule_generation(self, X, y, fuzzy_sets_per_input):
        """
        Generate rules from labeled data X, y.
        fuzzy_sets_per_input: list of lists of FuzzySet for each input feature.
        For each data point, create a rule using the max-membership fuzzy set per input.
        """
        self.rules = []
        for xi, yi in zip(X, y):
            antecedents = []
            for i, val in enumerate(xi):
                sets = fuzzy_sets_per_input[i]
                memberships = [fs.membership(val) for fs in sets]
                idx = int(np.argmax(memberships))
                antecedents.append((i, sets[idx]))
            rule = FuzzyRule(antecedents, yi)
            self.add_rule(rule)

    def evaluate(self, x):
        """
        Mamdani-style weighted average inference for regression/classification.
        Returns weighted sum of consequents by rule activation.
        """
        if not self.rules:
            return 0.0
        num = 0.0
        denom = 0.0
        for rule in self.rules:
            activation = 1.0
            for (i, fs) in rule.antecedents:
                activation *= fs.membership(x[i])
            num += activation * rule.consequent
            denom += activation
        return num / denom if denom > 0 else 0.0
