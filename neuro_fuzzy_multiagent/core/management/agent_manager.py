from neuro_fuzzy_multiagent.core.management.agent_factory import (
    create_agent_from_config,
)
from neuro_fuzzy_multiagent.core.management.message_bus import MessageBus


class AgentManager:
    """
    Manages dynamic addition and removal of agents at runtime.
    Integrates with MessageBus for plug-and-play communication.
    """

    def __init__(self, bus=None):
        self.bus = bus if bus is not None else MessageBus()
        self.agents = []
        self.agent_groups = {}

    def add_agent(self, config, group=None):
        agent = create_agent_from_config(config)
        self.bus.register(agent, group=group)
        self.agents.append(agent)
        if group:
            self.agent_groups.setdefault(group, []).append(agent)
        return agent

    def remove_agent(self, agent):
        self.bus.unregister(agent)
        if agent in self.agents:
            self.agents.remove(agent)
        for group, members in self.agent_groups.items():
            if agent in members:
                members.remove(agent)

    def reload_agent_config(self, agent, config_or_path):
        """
        Reload configuration for a specific agent at runtime.
        """
        agent.reload_config(config_or_path)

    def replace_agent(self, old_agent, new_config, group=None):
        """
        Replace an existing agent with a new one created from new_config.
        Optionally assign to the same group.
        Returns the new agent.
        """
        self.remove_agent(old_agent)
        return self.add_agent(new_config, group=group)

    def get_agents(self, group=None):
        if group:
            return list(self.agent_groups.get(group, []))
        return list(self.agents)
