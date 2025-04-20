from .base_actuator import BaseActuator

class DummyActuator(BaseActuator):
    """
    A minimal actuator plugin for plug-and-play testing. Prints the received command.
    """
    def write(self, command):
        print(f"DummyActuator received command: {command}")
