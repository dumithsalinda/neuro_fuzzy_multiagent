"""
Template Sensor Plugin
----------------------
Subclass BaseSensor and use the @register_plugin('sensor') decorator (already on base class).
"""

from neuro_fuzzy_multiagent.plugins.base_sensor import BaseSensor


class MyTemplateSensor(BaseSensor):
    """
    Example sensor for plug-and-play system.
    Configurable via dashboard/config.
    """

    def read(self):
        return 42
