class MultiAgentSystem:
    """
    Manages a group of agents and facilitates collaboration/communication.
    Supports broadcasting, direct messaging, coordinated actions, and group law enforcement.
    """
    def __init__(self, agents):
        self.agents = agents

    def broadcast(self, message, sender=None, group=None):
        """
        Broadcast a message to all agents except sender. If group is specified, only to group members.
        """
        for agent in self.agents:
            if agent is not sender:
                if group is None or (hasattr(agent, 'group') and agent.group == group):
                    agent.receive_message(message, sender=sender)

    def step_all(self, observations, states=None):
        """
        Step all agents with their respective observations (and optional states).
        Returns list of actions.
        """
        actions = []
        for i, agent in enumerate(self.agents):
            state = states[i] if states is not None else None
            actions.append(agent.act(observations[i], state=state))
        return actions

    def coordinate_actions(self, observations, states=None):
        """
        Let agents share their chosen actions, then agree on a consensus action (e.g., mean).
        Enforces group laws on the consensus action.
        Returns the consensus action if legal, else raises LawViolation.
        """
        actions = self.step_all(observations, states)
        import numpy as np
        consensus = np.mean(actions, axis=0)
        # Enforce group laws
        from core.laws import enforce_laws, LawViolation
        enforce_laws(consensus, state={'actions': actions}, category='group')
        return consensus

    def broadcast_knowledge(self, knowledge, sender=None):
        """
        Broadcast knowledge to all agents except sender, enforcing group and knowledge laws, respecting privacy.
        """
        from core.laws import enforce_laws, LawViolation
        privacy = knowledge.get('privacy', 'public') if isinstance(knowledge, dict) else 'public'
        try:
            enforce_laws(knowledge, state=None, category='group')
            enforce_laws(knowledge, state=None, category='knowledge')
            if privacy == 'private':
                return  # Do not share
            elif privacy == 'public':
                for agent in self.agents:
                    if agent is not sender:
                        agent.receive_message({'type': 'knowledge', 'content': knowledge}, sender=sender)
            elif privacy == 'group-only' and sender is not None and hasattr(sender, 'group'):
                for agent in self.agents:
                    if agent is not sender and hasattr(agent, 'group') and agent.group == sender.group:
                        agent.receive_message({'type': 'knowledge', 'content': knowledge, 'privacy': 'group-only', 'group': sender.group}, sender=sender)
            elif privacy == 'recipient-list' and 'recipients' in knowledge:
                for agent in knowledge['recipients']:
                    if agent is not sender:
                        agent.receive_message({'type': 'knowledge', 'content': knowledge, 'privacy': 'recipient-list', 'recipients': knowledge['recipients']}, sender=sender)
        except LawViolation as e:
            print(f"[MultiAgentSystem] Knowledge broadcast blocked by law: {e}")

    def aggregate_knowledge(self, attr='online_knowledge'):
        """
        Aggregate knowledge from all agents (e.g., for consensus or federated update).
        Returns a list of all non-None knowledge attributes.
        """
        return [getattr(agent, attr, None) for agent in self.agents if getattr(agent, attr, None) is not None]
