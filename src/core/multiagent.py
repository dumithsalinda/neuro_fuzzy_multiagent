class MultiAgentSystem:
    """
    Manages a group of agents and facilitates collaboration/communication.
    Supports broadcasting, direct messaging, coordinated actions, and group law enforcement.
    """
    def __init__(self, agents):
        self.agents = agents

    def broadcast(self, message, sender=None):
        """
        Broadcast a message to all agents except sender.
        """
        for agent in self.agents:
            if agent is not sender:
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

    def enforce_group_laws_on_knowledge(self, knowledge):
        """
        Enforce group laws on shared knowledge before accepting it.
        Raises LawViolation if any group law is broken.
        """
        from core.laws import enforce_laws
        enforce_laws(knowledge, state=None, category='group')
