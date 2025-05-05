# core/rules.py
"""
Global rule set enforcement for safety and policy in the neuro-fuzzy multiagent OS.
This module defines rules to ensure safe and ethical agent behavior.
"""
# Global rule set enforcement
class GlobalRules:
    """
    Enforces a set of global rules for agent actions.
    Used to ensure that agent behaviors comply with system-wide safety and policy constraints.
    """
    def __init__(self, rules):
        """
        Initialize with a list of rule functions.
        Args:
            rules (list): List of callable rule functions.
        """
        self.rules = rules

    def check(self, action):
        """
        Check if an action is allowed by all rules.
        Args:
            action (dict): The action to check (should have at least a 'type' key).
        Returns:
            bool: True if all rules allow the action, False otherwise.
        """
        for rule in self.rules:
            if not rule(action):
                return False
        return True


def never_delete_data(action):
    """
    Rule: Never allow actions that delete data.
    Args:
        action (dict): The action to check.
    Returns:
        bool: True if the action is not a data deletion, False otherwise.
    """
    return action.get("type") != "delete_data"
