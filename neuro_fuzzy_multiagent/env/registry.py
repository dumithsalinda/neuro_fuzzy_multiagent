"""
Environment Registry: Dynamically discovers and registers all subclasses of BaseEnvironment in src/env/.

Usage:
    from .registry import get_registered_environments
    envs = get_registered_environments()
    # envs is a dict: {class_name: class_obj}
"""

import logging

from neuro_fuzzy_multiagent.core.plugins.registration_utils import (
    get_registered_plugins,
    register_plugin,
)


def get_registered_environments():
    """
    Returns a dictionary {class_name: class_obj} of all registered environments.
    """
    try:
        return get_registered_plugins("environment")
    except Exception as e:
        logging.error(f"Failed to get registered environments: {e}")
        return {}


def get_registered_environments():
    """
    Returns a dictionary {class_name: class_obj} of all registered environments.
    """
    return dict(_REGISTRY)
