class MultiAgentSystem:
    """
    Manages a group of agents and facilitates collaboration/communication.
    Supports broadcasting, direct messaging, and coordinated actions.
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
        Example: Let agents share their chosen actions, then agree on a consensus action (e.g., mean).
        Returns the consensus action.
        """
        actions = self.step_all(observations, states)
        # For demonstration: take mean action as consensus
        import numpy as np
        return np.mean(actions, axis=0)
