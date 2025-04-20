"""
Plugin Registry: Dynamically discovers and registers all sensor and actuator plugins in src/plugins/.

Usage:
    from .registry import get_registered_sensors, get_registered_actuators
    sensors = get_registered_sensors()  # {class_name: class_obj}
    actuators = get_registered_actuators()  # {class_name: class_obj}
"""
import importlib
import pkgutil
import os
from .base_sensor import BaseSensor
from .base_actuator import BaseActuator

_SENSOR_REGISTRY = {}
_ACTUATOR_REGISTRY = {}

_plugins_dir = os.path.dirname(__file__)
for _, modname, _ in pkgutil.iter_modules([_plugins_dir]):
    if modname in ("registry", "__init__", "base_sensor", "base_actuator"): continue
    module = importlib.import_module(f"src.plugins.{modname}")
    for attr in dir(module):
        obj = getattr(module, attr)
        if isinstance(obj, type):
            if issubclass(obj, BaseSensor) and obj is not BaseSensor:
                _SENSOR_REGISTRY[obj.__name__] = obj
            if issubclass(obj, BaseActuator) and obj is not BaseActuator:
                _ACTUATOR_REGISTRY[obj.__name__] = obj

def get_registered_sensors():
    """
    Returns a dictionary {class_name: class_obj} of all registered sensor plugins.
    """
    return dict(_SENSOR_REGISTRY)

def get_registered_actuators():
    """
    Returns a dictionary {class_name: class_obj} of all registered actuator plugins.
    """
    return dict(_ACTUATOR_REGISTRY)
