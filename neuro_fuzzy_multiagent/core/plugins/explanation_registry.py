"""
Explanation function registry for agent plugins.
Allows custom explanation functions to be registered for agent types.
"""

_explanation_registry = {}


def register_explanation(agent_cls):
    """Decorator to register a custom explanation function for an agent class."""

    def decorator(func):
        _explanation_registry[agent_cls] = func
        return func

    return decorator


def get_explanation(agent_instance, *args, **kwargs):
    """Call the registered explanation function for the agent, if any."""
    func = _explanation_registry.get(type(agent_instance), None)
    if func is not None:
        return func(agent_instance, *args, **kwargs)
    # fallback to agent's own explain_action if present
    if hasattr(agent_instance, "explain_action"):
        return agent_instance.explain_action(*args, **kwargs)
    raise NotImplementedError(
        f"No explanation function registered for {type(agent_instance).__name__}"
    )


def clear_explanations():
    _explanation_registry.clear()
