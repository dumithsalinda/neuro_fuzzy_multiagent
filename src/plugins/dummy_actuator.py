from .base_actuator import BaseActuator

from src.core.plugins.registration_utils import register_plugin

@register_plugin('actuator')
class DummyActuator(BaseActuator):
    """
    A minimal actuator plugin for plug-and-play testing. Prints the received command.
    """
    def write(self, command):
        print(f"DummyActuator received command: {command}")
