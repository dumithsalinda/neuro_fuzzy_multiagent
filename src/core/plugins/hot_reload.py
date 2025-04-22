"""
hot_reload.py

Utility functions to clear and reload all plugin registries and modules at runtime.
"""
import importlib
import logging
import sys
from src.core.plugins.registration_utils import clear_plugin_registry

PLUGIN_MODULE_PATHS = [
    'src.plugins',
    'src.env',
    'src.core.agents',
]
PLUGIN_TYPES = ['sensor', 'actuator', 'environment', 'agent']


def reload_all_plugins():
    """
    Clear all plugin registries and reload all plugin modules.
    Returns a dict with status and errors (if any).
    """
    result = {'reloaded': [], 'errors': []}
    # Clear registries
    for ptype in PLUGIN_TYPES:
        clear_plugin_registry(ptype)
    # Reload modules
    for module_path in PLUGIN_MODULE_PATHS:
        try:
            if module_path in sys.modules:
                importlib.reload(sys.modules[module_path])
            else:
                importlib.import_module(module_path)
            result['reloaded'].append(module_path)
        except Exception as e:
            logging.error(f"Failed to reload {module_path}: {e}")
            result['errors'].append((module_path, str(e)))
    return result
