import numpy as np
import pytest

from neuro_fuzzy_multiagent.core.agents.agent import Agent
from neuro_fuzzy_multiagent.core.management.multiagent import MultiAgentSystem


class DummyAgent(Agent):
    def __init__(self, model=None):
        super().__init__(model)
        self.received = []

    def receive_message(self, message, sender=None):
        self.received.append((message, sender))


def test_broadcast_message():
    agents = [DummyAgent() for _ in range(3)]
    system = MultiAgentSystem(agents)
    system.broadcast("hello", sender=agents[0])
    assert agents[0].received == []
    assert all(("hello", agents[0]) in a.received for a in agents[1:])


def test_coordinate_actions_mean():
    # Each agent returns a fixed action
    class FixedAgent(Agent):
        def __init__(self, action):
            super().__init__(None)
            self._action = np.array([action])

        def act(self, observation, state=None):
            return self._action

    agents = [FixedAgent(0), FixedAgent(2), FixedAgent(4)]
    system = MultiAgentSystem(agents)
    consensus = system.coordinate_actions([None, None, None])
    assert np.allclose(consensus, np.mean([0, 2, 4]))
