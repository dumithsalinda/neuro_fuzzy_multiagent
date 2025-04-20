from .base_sensor import BaseSensor

class DummySensor(BaseSensor):
    """
    A minimal sensor plugin for plug-and-play testing. Returns a constant value.
    """
    def read(self):
        return 42
