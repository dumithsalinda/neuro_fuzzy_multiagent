import pytest

from neuro_fuzzy_multiagent.core.agents.agent import Agent
from neuro_fuzzy_multiagent.core.management.multiagent_system import MultiAgentSystem


class DummyAgent(Agent):
    def __init__(self):
        super().__init__(model=None)


# --- Dynamic Agent Management Tests ---
def test_add_and_remove_agent():
    agents = [DummyAgent() for _ in range(2)]
    mas = MultiAgentSystem(agents)
    new_agent = DummyAgent()
    mas.add_agent(new_agent)
    assert new_agent in mas.agents
    assert len(mas.agents) == 3
    mas.remove_agent(new_agent)
    assert new_agent not in mas.agents
    assert len(mas.agents) == 2
    # Remove by index
    mas.remove_agent(0)
    assert len(mas.agents) == 1


def test_group_creation_and_movement():
    agents = [DummyAgent() for _ in range(4)]
    mas = MultiAgentSystem(agents)
    mas.create_group("A", [0, 1])
    mas.create_group("B", [2, 3])
    assert mas.groups["A"] == {0, 1}
    assert mas.groups["B"] == {2, 3}
    # Move agent from A to B
    mas.move_agent_to_group(1, "B")
    assert 1 not in mas.groups["A"]
    assert 1 in mas.groups["B"]
    # Dissolve group
    mas.dissolve_group("A")
    assert "A" not in mas.groups


def test_group_leader_tracking():
    agents = [DummyAgent() for _ in range(3)]
    mas = MultiAgentSystem(agents)
    mas.create_group("X", [0, 2])
    assert mas.group_leaders["X"] == 0
    mas.move_agent_to_group(2, "X")
    assert mas.group_leaders["X"] == 0  # Leader remains unless removed
    mas.remove_agent(0)
    # After removing leader, leader entry should be deleted
    assert "X" not in mas.group_leaders or mas.group_leaders["X"] != 0
