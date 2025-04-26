from neuro_fuzzy_multiagent.core.plugins.registration_utils import register_plugin
from neuro_fuzzy_multiagent.plugins.base_actuator import BaseActuator


@register_plugin("actuator")
class DummyActuator(BaseActuator):
    """
    A minimal actuator plugin for plug-and-play testing. Prints the received command.
    """

    def write(self, command):
        print(f"DummyActuator received command: {command}")
