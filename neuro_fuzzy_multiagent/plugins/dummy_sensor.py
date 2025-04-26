from neuro_fuzzy_multiagent.core.plugins.registration_utils import register_plugin
from neuro_fuzzy_multiagent.plugins.base_sensor import BaseSensor


@register_plugin("sensor")
class DummySensor(BaseSensor):
    """
    A minimal sensor plugin for plug-and-play testing. Returns a constant value.
    """

    def read(self):
        return 42
