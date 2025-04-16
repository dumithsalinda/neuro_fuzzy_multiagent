# core/rules.py
# Global rule set enforcement
class GlobalRules:
    def __init__(self, rules):
        self.rules = rules
    def check(self, action):
        # Return True if action is allowed, False otherwise
        for rule in self.rules:
            if not rule(action):
                return False
        return True
# Example: Never delete data
def never_delete_data(action):
    return action.get('type') != 'delete_data'
