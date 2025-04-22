from core.plugins.registration_utils import register_plugin

@register_plugin('actuator')
class ExampleActuator:
    """
    Example actuator plugin that sets and stores the last action.
    """
    def __init__(self):
        self.last_action = None

    def actuate(self, action):
        """Perform the given action (store it for demo purposes)."""
        self.last_action = action
        return f"Action {action} performed."
