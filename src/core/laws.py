"""
laws.py

Defines and enforces unbreakable laws (hard constraints) for agent actions.
"""

class LawViolation(Exception):
    pass

# Example: list of law-checking functions
_laws = []

def register_law(law_fn):
    """Register a new law-checking function."""
    _laws.append(law_fn)
    return law_fn

def clear_laws():
    """Clear all registered laws (for testing/extending)."""
    _laws.clear()

def enforce_laws(action, state=None):
    """
    Check all registered laws. Raise LawViolation if any are broken.
    """
    for law in _laws:
        if not law(action, state):
            raise LawViolation(f"Law violated: {law.__name__}")

# --- Example default laws ---
@register_law
def action_is_finite(action, state=None):
    """All action values must be finite numbers."""
    import numpy as np
    return np.all(np.isfinite(action))

@register_law
def action_within_bounds(action, state=None):
    """All action values must be in [-10, 10] (example bound)."""
    import numpy as np
    return np.all((action >= -10) & (action <= 10))
