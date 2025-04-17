class MessageBus:
    """
    General-purpose message bus for agent communication.
    Supports broadcast, direct, and group messaging.
    """
    def __init__(self):
        self.agents = []
        self.groups = {}

    def register(self, agent, group=None):
        if agent not in self.agents:
            self.agents.append(agent)
        if group is not None:
            self.groups.setdefault(group, []).append(agent)

    def unregister(self, agent):
        if agent in self.agents:
            self.agents.remove(agent)
        for group in self.groups.values():
            if agent in group:
                group.remove(agent)

    def send(self, message, recipient):
        recipient.receive_message(message)

    def broadcast(self, message, sender=None):
        for agent in self.agents:
            if agent is not sender:
                agent.receive_message(message, sender=sender)

    def groupcast(self, message, group, sender=None):
        for agent in self.groups.get(group, []):
            if agent is not sender:
                agent.receive_message(message, sender=sender)
