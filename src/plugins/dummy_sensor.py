from .base_sensor import BaseSensor

from src.core.plugins.registration_utils import register_plugin

@register_plugin('sensor')
class DummySensor(BaseSensor):
    """
    A minimal sensor plugin for plug-and-play testing. Returns a constant value.
    """
    def read(self):
        return 42
