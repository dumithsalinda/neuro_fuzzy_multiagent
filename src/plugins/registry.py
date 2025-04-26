"""
Plugin Registry: Dynamically discovers and registers all sensor and actuator plugins in src/plugins/.

Usage:
    from .registry import get_registered_sensors, get_registered_actuators
    sensors = get_registered_sensors()  # {class_name: class_obj}
    actuators = get_registered_actuators()  # {class_name: class_obj}
"""

import logging
from src.core.plugins.registration_utils import get_registered_plugins, register_plugin


def get_registered_sensors():
    """
    Returns a dictionary {class_name: class_obj} of all registered sensor plugins.
    """
    try:
        return get_registered_plugins("sensor")
    except Exception as e:
        logging.error(f"Failed to get registered sensors: {e}")
        return {}


def get_registered_actuators():
    """
    Returns a dictionary {class_name: class_obj} of all registered actuator plugins.
    """
    try:
        return get_registered_plugins("actuator")
    except Exception as e:
        logging.error(f"Failed to get registered actuators: {e}")
        return {}


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
