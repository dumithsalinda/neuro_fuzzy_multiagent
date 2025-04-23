from src.core.plugins.registration_utils import register_plugin

@register_plugin('sensor')
class ExampleSensor:
    """
    Example sensor plugin that returns a fixed observation.
    """
    def __init__(self, observation=0):
        self.observation = observation

    def read(self):
        """Return the current observation."""
        return self.observation
