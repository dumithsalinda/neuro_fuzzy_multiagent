"""
fuzzy_system.py

Implements fuzzy set, fuzzy rule, and fuzzy inference system classes for neuro-fuzzy and hybrid learning systems.
Supports dynamic rule generation, self-tuning membership functions, and interpretable fuzzy logic modeling.

Classes:
- FuzzySet: Represents a fuzzy set with a Gaussian membership function and self-tuning capability.
- FuzzyRule: Encodes an IF-THEN rule with antecedents and a consequent.
- FuzzyInferenceSystem: Manages rules and performs fuzzy inference for reasoning and learning.

This module enables interpretable, adaptive, and robust fuzzy logic-based reasoning in multiagent and hybrid systems.
"""

import numpy as np

class FuzzySet:
    """
    Represents a fuzzy set with a membership function.
    Supports self-tuning of membership parameters.

    Parameters
    ----------
    name : str
        Name of the fuzzy set (e.g., 'Low', 'High').
    params : list or np.ndarray
        Parameters for the membership function (e.g., [center, width] for Gaussian).
    """
    def __init__(self, name, params):
        self.name = name
        self.params = params  # e.g., [center, width]

    def membership(self, x):
        """
        Gaussian membership function.

        Parameters
        ----------
        x : float or np.ndarray
            Input value(s) to evaluate membership.
        Returns
        -------
        float or np.ndarray
            Membership grade(s) in [0, 1].
        """
        c, w = self.params
        return np.exp(-0.5 * ((x - c) / w) ** 2)

    def tune(self, data):
        """
        Self-tune the center and width using data (mean/std update).

        Parameters
        ----------
        data : array-like
            Samples belonging to this fuzzy set.
        Updates the internal parameters to fit the data.
        """
        if len(data) > 0:
            c = np.mean(data)
            w = np.std(data) + 1e-6  # avoid zero std
            self.params = [c, w]

class FuzzyRule:
    """
    Represents a fuzzy rule (IF-THEN) with antecedents and a consequent.

    Parameters
    ----------
    antecedents : list of (int, FuzzySet)
        List of (input_index, FuzzySet) pairs for the rule antecedent.
    consequent : float or object
        Output value or class label for the rule's THEN part.
    """
    def __init__(self, antecedents, consequent):
        self.antecedents = antecedents  # list of (input_index, FuzzySet)
        self.consequent = consequent    # output value

class FuzzyInferenceSystem:
    """
    Fuzzy inference system with dynamic rule generation and evaluation.
    Manages fuzzy rules and computes outputs via fuzzy logic reasoning.
    Supports immutable core rules and append-only dynamic rules for safety.
    """
    def __init__(self, core_rules=None):
        """
        Initialize the fuzzy inference system.
        core_rules: list of FuzzyRule, immutable base rule set (optional)
        """
        self.core_rules = core_rules if core_rules is not None else []
        self.dynamic_rules = []  # rules from feedback/robustness
        self.rules = self.core_rules + self.dynamic_rules

    def _is_duplicate_of_core(self, new_rule):
        for rule in self.core_rules:
            if ([(i, fs.name) for i, fs in rule.antecedents] == [(i, fs.name) for i, fs in new_rule.antecedents]) and rule.consequent == new_rule.consequent:
                return True
        return False

    def add_rule_from_feedback(self, antecedent_values, consequent, fuzzy_sets_per_input):
        """
        Add a fuzzy rule based on human feedback.
        Does NOT override core rules. Warns if duplicate.
        """
        antecedents = []
        for i, val in enumerate(antecedent_values):
            sets = fuzzy_sets_per_input[i]
            memberships = [fs.membership(val) for fs in sets]
            idx = int(np.argmax(memberships))
            antecedents.append((i, sets[idx]))
        rule = FuzzyRule(antecedents, consequent)
        if self._is_duplicate_of_core(rule):
            print("[Warning] Attempted to add a rule that duplicates an immutable core rule. Ignoring.")
            return False
        self.dynamic_rules.append(rule)
        self.rules = self.core_rules + self.dynamic_rules
        return True

    def add_rule(self, rule, as_core=False):
        """
        Add a fuzzy rule to the system.
        If as_core=True, adds to immutable core rules. Otherwise, to dynamic rules.
        """
        if as_core:
            self.core_rules.append(rule)
        else:
            if self._is_duplicate_of_core(rule):
                print("[Warning] Attempted to add a rule that duplicates an immutable core rule. Ignoring.")
                return False
            self.dynamic_rules.append(rule)
        self.rules = self.core_rules + self.dynamic_rules
        return True

    def get_core_rules(self):
        return self.core_rules

    def get_dynamic_rules(self):
        return self.dynamic_rules

    def add_rule_from_feedback(self, antecedent_values, consequent, fuzzy_sets_per_input):
        """
        Add a fuzzy rule based on human feedback.
        antecedent_values: list of floats (input values for the rule antecedents)
        consequent: float or int (desired output/action)
        fuzzy_sets_per_input: list of lists of FuzzySet for each input feature
        """
        antecedents = []
        for i, val in enumerate(antecedent_values):
            sets = fuzzy_sets_per_input[i]
            memberships = [fs.membership(val) for fs in sets]
            idx = int(np.argmax(memberships))
            antecedents.append((i, sets[idx]))
        rule = FuzzyRule(antecedents, consequent)
        self.add_rule(rule)

    def add_rule(self, rule):
        """
        Add a fuzzy rule to the system.

        Parameters
        ----------
        rule : FuzzyRule
            Rule to add to the system.
        """
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
