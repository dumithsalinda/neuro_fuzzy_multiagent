"""
laws.py

Defines and enforces unbreakable laws (hard constraints) for agent actions.
"""

class LawViolation(Exception):
    pass

# Laws are now stored by category (default: 'action')
_LAWS = {'action': []}

def register_law(law_fn, category='action'):
    """Register a new law-checking function under a category."""
    if category not in _LAWS:
        _LAWS[category] = []
    _LAWS[category].append(law_fn)
    return law_fn

def remove_law(law_fn, category='action'):
    """Remove a law-checking function from a category."""
    if category in _LAWS and law_fn in _LAWS[category]:
        _LAWS[category].remove(law_fn)

def clear_laws(category=None):
    """Clear all laws in a category or all categories if None."""
    if category:
        _LAWS[category] = []
    else:
        for cat in _LAWS:
            _LAWS[cat] = []

def list_laws(category=None):
    """List all registered laws, optionally by category."""
    if category:
        return list(_LAWS.get(category, []))
    return {cat: list(laws) for cat, laws in _LAWS.items()}

def enforce_laws(action, state=None, category='action'):
    """
    Check all registered laws in a category. Raise LawViolation if any are broken.
    """
    for law in _LAWS.get(category, []):
        if not law(action, state):
            raise LawViolation(f"Law violated in {category}: {law.__name__}")

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
