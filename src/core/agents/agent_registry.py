"""
Agent Registry: Dynamically discovers and registers all agent classes in src/core/.

Usage:
    from src.core.agents.agent_registry import get_registered_agents
    agents = get_registered_agents()
    # agents is a dict: {class_name: class_obj}
"""
import importlib
import pkgutil
import os
from src.core.agents.agent import Agent

_REGISTRY = {}

# Discover all .py files in this directory (except agent_registry.py and __init__.py)
_core_dir = os.path.dirname(__file__)
for _, modname, _ in pkgutil.iter_modules([_core_dir]):
    if modname in ("agent_registry", "__init__"): continue
    module = importlib.import_module(f"src.core.agents.{modname}")
    for attr in dir(module):
        obj = getattr(module, attr)
        if (
            isinstance(obj, type)
            and (obj is Agent or (issubclass(obj, Agent) and obj is not Agent))
        ):
            _REGISTRY[obj.__name__] = obj

def get_registered_agents():
    """
    Returns a dictionary {class_name: class_obj} of all registered agent classes.
    """
    return dict(_REGISTRY)
