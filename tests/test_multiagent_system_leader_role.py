import pytest
from src.core.multiagent_system import MultiAgentSystem
from src.core.agent import Agent

class DummyAgent(Agent):
    def __init__(self):
        super().__init__(model=None)

def test_leader_election_default_and_custom():
    agents = [DummyAgent() for _ in range(4)]
    mas = MultiAgentSystem(agents)
    mas.create_group('G', [2, 1, 3])
    # Default: lowest index
    leader = mas.elect_leader('G')
    assert leader == 1
    assert mas.get_leader('G') == 1
    assert mas.get_role('G', 1) == 'leader'
    # Custom: highest index
    leader2 = mas.elect_leader('G', election_fn=max)
    assert leader2 == 3
    assert mas.get_leader('G') == 3
    assert mas.get_role('G', 3) == 'leader'

def test_role_assignment_and_query():
    agents = [DummyAgent() for _ in range(3)]
    mas = MultiAgentSystem(agents)
    mas.create_group('A', [0, 1, 2])
    mas.assign_role('A', 1, 'scout')
    mas.assign_role('A', 2, 'explorer')
    assert mas.get_role('A', 1) == 'scout'
    assert mas.get_role('A', 2) == 'explorer'
    assert mas.get_role('A', 0) is None
    # Leader election updates role
    mas.elect_leader('A', election_fn=max)
    leader = mas.get_leader('A')
    assert mas.get_role('A', leader) == 'leader'

def test_leader_removed_from_group():
    agents = [DummyAgent() for _ in range(2)]
    mas = MultiAgentSystem(agents)
    mas.create_group('B', [0, 1])
    mas.elect_leader('B')
    assert mas.get_leader('B') == 0
    mas.remove_agent(0)
    # After removal, only one agent remains at index 0
    assert mas.get_leader('B') == 0
    # Can re-elect, should still be index 0
    mas.elect_leader('B')
    assert mas.get_leader('B') == 0
