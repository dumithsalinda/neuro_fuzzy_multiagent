from abc import ABC, abstractmethod

from neuro_fuzzy_multiagent.core.plugins.registration_utils import register_plugin


@register_plugin("sensor")
class BaseSensor(ABC):
    """
    Base class for all sensor plugins. Sensors provide observations or data streams to environments or agents.
    """

    @abstractmethod
    def read(self):
        """Return the latest sensor reading (any type)."""
        pass
