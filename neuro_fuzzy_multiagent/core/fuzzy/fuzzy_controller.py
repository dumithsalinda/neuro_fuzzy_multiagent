import yaml

class FuzzyController:
    def __init__(self, rules=None):
        self.rules = rules or {}

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            rules = yaml.safe_load(f)
        return cls(rules)

    def explain(self, input, output=None):
        # Placeholder: return a simple explanation
        return f"Fuzzy explanation based on input: {input} and output: {output} using rules: {self.rules}"
