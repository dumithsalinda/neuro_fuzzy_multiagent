"""
Agent Registry: Dynamically discovers and registers all agent classes in src/core/.

Usage:
    from src.core.agents.agent_registry import get_registered_agents
    agents = get_registered_agents()
    # agents is a dict: {class_name: class_obj}
"""

import logging
from src.core.plugins.registration_utils import get_registered_plugins, register_plugin


def get_registered_agents():
    """
    Returns a dictionary {class_name: class_obj} of all registered agent classes.
    """
    try:
        return get_registered_plugins("agent")
    except Exception as e:
        logging.error(f"Failed to get registered agents: {e}")
        return {}


def get_registered_agents():
    """
    Returns a dictionary {class_name: class_obj} of all registered agent classes.
    """
    return dict(_REGISTRY)
