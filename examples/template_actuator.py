"""
Template Actuator Plugin
------------------------
Subclass BaseActuator and use the @register_plugin('actuator') decorator (already on base class).
"""

from neuro_fuzzy_multiagent.plugins.base_actuator import BaseActuator


class MyTemplateActuator(BaseActuator):
    """
    Example actuator for plug-and-play system.
    Configurable via dashboard/config.
    """

    def write(self, command):
        print(f"Received command: {command}")
