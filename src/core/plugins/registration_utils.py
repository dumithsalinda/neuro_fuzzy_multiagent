"""
registration_utils.py

Provides a standardized decorator for plugin registration across agents, environments, sensors, and actuators.
"""

import logging

# Global registries for each plugin type
PLUGIN_REGISTRIES = {
    'agent': {},
    'environment': {},
    'sensor': {},
    'actuator': {},
}


def register_plugin(plugin_type):
    """
    Decorator to register a class as a plugin of the given type.
    Usage: @register_plugin('agent')
    """
    def decorator(cls):
        name = getattr(cls, '__plugin_name__', cls.__name__)
        if name in PLUGIN_REGISTRIES[plugin_type]:
            logging.warning(f"Duplicate plugin registration for {plugin_type}: {name}")
        PLUGIN_REGISTRIES[plugin_type][name] = cls
        return cls
    return decorator


def get_registered_plugins(plugin_type):
    """
    Returns a dict {name: class} for the given plugin type.
    """
    return dict(PLUGIN_REGISTRIES[plugin_type])


def clear_plugin_registry(plugin_type):
    PLUGIN_REGISTRIES[plugin_type].clear()
