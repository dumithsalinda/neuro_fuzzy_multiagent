"""
universal_fuzzy_layer.py

A universal fuzzy logic layer that can be attached to any agent type (classic, neural, RL, etc.).
Supports dynamic rule management, fuzzy set configuration, and explainability.
"""

from .fuzzy_system import FuzzyInferenceSystem, FuzzyRule


class UniversalFuzzyLayer:
    """
    Universal fuzzy logic layer for plug-and-play integration with any agent.
    """

    def __init__(self, fuzzy_sets_per_input=None, core_rules=None):
        self.fis = FuzzyInferenceSystem(core_rules=core_rules)
        self.fuzzy_sets_per_input = fuzzy_sets_per_input or []

    def evaluate(self, x):
        """
        Evaluate the fuzzy inference system on input x.
        """
        return self.fis.evaluate(x)

    def add_rule(self, antecedents, consequent, as_core=False):
        """
        Add a fuzzy rule at runtime.
        antecedents: list of (input_index, FuzzySet)
        consequent: output value or class label
        as_core: if True, add as core rule; otherwise, as dynamic rule
        """
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

    def explain(self, x):
        """
        Return rule activations and inference trace for input x.
        """
        rule_activations = []
        for rule in self.fis.rules:
            activation = 1.0
            for i, fs in rule.antecedents:
                activation *= fs.membership(x[i])
            rule_activations.append(activation)
        output = self.evaluate(x)
        return {"rule_activations": rule_activations, "output": output}
