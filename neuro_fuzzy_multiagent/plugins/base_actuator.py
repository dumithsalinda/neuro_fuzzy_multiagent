from abc import ABC, abstractmethod

from neuro_fuzzy_multiagent.core.plugins.registration_utils import register_plugin


@register_plugin("actuator")
class BaseActuator(ABC):
    """
    Base class for all actuator plugins. Actuators receive actions/commands from agents or environments and execute them in the real or simulated world.
    """

    @abstractmethod
    def write(self, command):
        """Execute a command (any type)."""
        pass
