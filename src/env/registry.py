"""
Environment Registry: Dynamically discovers and registers all subclasses of BaseEnvironment in src/env/.

Usage:
    from .registry import get_registered_environments
    envs = get_registered_environments()
    # envs is a dict: {class_name: class_obj}
"""
import importlib
import pkgutil
import os
from .base_env import BaseEnvironment

_REGISTRY = {}

# Discover all .py files in this directory (except registry.py and __init__.py)
_env_dir = os.path.dirname(__file__)
for _, modname, _ in pkgutil.iter_modules([_env_dir]):
    if modname in ("registry", "__init__"): continue
    module = importlib.import_module(f"src.env.{modname}")
    for attr in dir(module):
        obj = getattr(module, attr)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseEnvironment)
            and obj is not BaseEnvironment
        ):
            _REGISTRY[obj.__name__] = obj

def get_registered_environments():
    """
    Returns a dictionary {class_name: class_obj} of all registered environments.
    """
    return dict(_REGISTRY)
