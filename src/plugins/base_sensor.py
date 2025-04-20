from abc import ABC, abstractmethod

class BaseSensor(ABC):
    """
    Base class for all sensor plugins. Sensors provide observations or data streams to environments or agents.
    """
    @abstractmethod
    def read(self):
        """Return the latest sensor reading (any type)."""
        pass
