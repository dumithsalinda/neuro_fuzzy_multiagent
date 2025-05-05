import yaml

class GlobalRules:
    def __init__(self, rules=None):
        self.rules = rules or {}

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            rules = yaml.safe_load(f)
        return cls(rules)

    def is_action_safe(self, action, context=None):
        # Placeholder: always safe, with dummy reason
        return True, "No real safety checks implemented (stub)."

    def check(self, action, context=None):
        # For compatibility with OS agent usage
        return self.is_action_safe(action, context)
