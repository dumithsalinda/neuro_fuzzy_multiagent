from abc import ABC, abstractmethod

class BaseActuator(ABC):
    """
    Base class for all actuator plugins. Actuators receive actions/commands from agents or environments and execute them in the real or simulated world.
    """
    @abstractmethod
    def write(self, command):
        """Execute a command (any type)."""
        pass
