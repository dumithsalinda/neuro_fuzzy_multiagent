"""
Template Agent Plugin
---------------------
Subclass Agent and use the @register_plugin('agent') decorator (already on base class).
"""

from src.core.agents.agent import Agent


class MyTemplateAgent(Agent):
    """
    Example agent for plug-and-play system.
    Configurable via dashboard/config.
    """

    def __init__(self, model=None, policy=None):
        super().__init__(model, policy)

    def act(self, observation):
        # Example: always return 0
        return 0
