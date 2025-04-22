"""
registration_utils.py

Provides a standardized decorator for plugin registration across agents, environments, sensors, and actuators.
"""

from typing import Callable, Dict, Any
import logging

# Global registries for each plugin type
PLUGIN_REGISTRIES = {
    'agent': {},
    'environment': {},
    'sensor': {},
    'actuator': {},
}


def register_plugin(plugin_type: str) -> Callable:
    """
    Decorator to register a class as a plugin of the given type.
    Usage: @register_plugin('agent')

    Args:
        plugin_type (str): The type of plugin to register.

    Returns:
        Callable: A decorator function that registers the plugin.
    """
    def decorator(cls: type) -> type:
        """
        Registers the plugin class.

        Args:
            cls (type): The plugin class to register.

        Returns:
            type: The registered plugin class.
        """
        name = getattr(cls, '__plugin_name__', cls.__name__)
        if name in PLUGIN_REGISTRIES[plugin_type]:
            logging.warning(f"Duplicate plugin registration for {plugin_type}: {name}")
        PLUGIN_REGISTRIES[plugin_type][name] = cls
        logging.info(f"Registered plugin: {plugin_type}.{name}")
        return cls
    return decorator


def get_registered_plugins(plugin_type: str) -> Dict[str, Any]:
    """
    Returns a dict {name: class} for the given plugin type.

    Args:
        plugin_type (str): The type of plugin to retrieve.

    Returns:
        Dict[str, Any]: A dictionary of registered plugins for the given type.
    """
    return dict(PLUGIN_REGISTRIES[plugin_type])


def clear_plugin_registry(plugin_type: str) -> None:
    """
    Clear the registry for a specific plugin type.

    Args:
        plugin_type (str): The type of plugin to clear.

    Returns:
        None
    """
    PLUGIN_REGISTRIES[plugin_type].clear()
    logging.info(f"Cleared plugin registry for type: {plugin_type}")
