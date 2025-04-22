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


import os
from pathlib import Path

from typing import Dict, Any
import logging

def reload_all_plugins() -> Dict[str, Any]:
    """
    Clear all plugin registries and reload all plugin modules and all plugin files.
    Returns a dict with status and errors (if any).
    """
    result = {'reloaded': [], 'errors': []}
    # Clear registries
    for ptype in PLUGIN_TYPES:
        clear_plugin_registry(ptype)
    # Reload top-level modules
    for module_path in PLUGIN_MODULE_PATHS:
        try:
            if module_path in sys.modules:
                importlib.reload(sys.modules[module_path])
            else:
                importlib.import_module(module_path)
            result['reloaded'].append(module_path)
            logging.info(f"Reloaded plugin module: {module_path}")
        except Exception as e:
            logging.error(f"Failed to reload module {module_path}: {e}")
            result['errors'].append((module_path, str(e)))
        except Exception as e:
            logging.error(f"Failed to reload {module_path}: {e}")
            result['errors'].append((module_path, str(e)))
    # Dynamically import all plugin files to trigger registration
    plugin_dirs = [
        ("src/plugins", "src.plugins"),
        ("src/env", "src.env"),
        ("src/core/agents", "src.core.agents"),
    ]
    for dir_path, pkg_base in plugin_dirs:
        for fname in os.listdir(dir_path):
            if not fname.endswith(".py") or fname.startswith("__init__") or fname.startswith("base_"):
                continue
            mod_name = fname[:-3]
            module_path = f"{pkg_base}.{mod_name}"
            try:
                if module_path in sys.modules:
                    importlib.reload(sys.modules[module_path])
                else:
                    importlib.import_module(module_path)
                result['reloaded'].append(module_path)
            except Exception as e:
                logging.error(f"Failed to import plugin file {module_path}: {e}")
                result['errors'].append((module_path, str(e)))
    return result

