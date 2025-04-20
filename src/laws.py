# Laws Definition File

"""
This module stores and manages all unbreakable laws that govern agent behavior.
Laws can be categorized as:
- Government Laws (legal/ethical constraints)
- Organization Laws (company, team, or project-specific rules)
- Personal Laws (user-defined preferences)

Functions are provided to register, remove, list, and enforce laws.
"""

from typing import Callable, Dict, List, Any

# Each law is a callable: law(action, state) -> bool
LAWS: Dict[str, List[Callable[[Any, Any], bool]]] = {
    "government": [],
    "organization": [],
    "personal": [],
}


# Example law registration (can be removed or replaced)
def example_law_no_negative_action(action, state):
    """Disallow negative actions (example)."""
    return action >= 0


# API


def register_law(category: str, law_fn: Callable[[Any, Any], bool]):
    """Register a new law in the specified category."""
    if category not in LAWS:
        raise ValueError(f"Unknown law category: {category}")
    LAWS[category].append(law_fn)


def remove_law(category: str, law_fn: Callable[[Any, Any], bool]):
    """Remove a law from the specified category."""
    if category in LAWS and law_fn in LAWS[category]:
        LAWS[category].remove(law_fn)


# Default allow-all law


def allow_all(action, state):
    """Allow all actions (default law)."""
    return True


def clear_laws():
    """Remove all laws from all categories (for testing/extending). Adds a default allow-all law to each category."""
    for cat in LAWS:
        LAWS[cat].clear()
        LAWS[cat].append(allow_all)


def list_laws(category: str) -> List[Callable[[Any, Any], bool]]:
    """List all laws in the specified category."""
    return LAWS.get(category, [])


def enforce_laws(action, state, categories=None):
    """
    Enforce all laws in the specified categories (default: all).
    Raises Exception if any law is violated.
    """
    if categories is None:
        categories = LAWS.keys()
    for cat in categories:
        if not LAWS[cat]:
            continue  # No laws in this category, skip
        for law in LAWS[cat]:
            if not law(action, state):
                doc = (
                    law.__doc__ or getattr(law, "__name__", str(law)) or "Law violated"
                )
                raise Exception(f"Unbreakable law violated in category '{cat}': {doc}")
    return True


# Example usage (remove in production):
# register_law('personal', example_law_no_negative_action)
# enforce_laws(-1, state={})  # Raises Exception
