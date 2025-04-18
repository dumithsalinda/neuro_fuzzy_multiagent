import numpy as np
import pytest
from src.core.laws import register_law, remove_law, clear_laws, list_laws, enforce_laws, LawViolation
from src.core.multiagent import MultiAgentSystem
from src.core.agent import Agent

def test_register_and_list_laws():
    clear_laws('group')
    def group_law(x, state=None):
        return np.all(x < 5)
    register_law(group_law, category='group')
    laws = list_laws('group')
    assert group_law in laws
    remove_law(group_law, category='group')
    assert group_law not in list_laws('group')

def test_enforce_group_law_blocks_violation():
    clear_laws('group')
    def block_large_action(x, state=None):
        return np.all(x < 2)
    register_law(block_large_action, category='group')
    agents = [Agent(model=None) for _ in range(3)]
    system = MultiAgentSystem(agents)
    # Patch step_all to return large actions
    system.step_all = lambda obs, states=None: [np.array([3]), np.array([3]), np.array([3])]
    with pytest.raises(LawViolation):
        system.coordinate_actions([None, None, None])

def test_group_law_allows_legal_consensus():
    clear_laws('group')
    def allow_small(x, state=None):
        return np.all(x < 10)
    register_law(allow_small, category='group')
    agents = [Agent(model=None) for _ in range(2)]
    system = MultiAgentSystem(agents)
    system.step_all = lambda obs, states=None: [np.array([1]), np.array([2])]
    consensus = system.coordinate_actions([None, None])
    assert np.all(consensus < 10)
