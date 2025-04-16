import numpy as np
import pytest
from core.multiagent import MultiAgentSystem
from core.agent import Agent
from core.laws import clear_laws, register_law, LawViolation

def test_group_decision_mean():
    agents = [Agent(model=None) for _ in range(3)]
    system = MultiAgentSystem(agents)
    system.step_all = lambda obs, states=None: [np.array([1]), np.array([2]), np.array([3])]
    result = system.group_decision([None, None, None], method="mean")
    assert np.allclose(result, [2])

def test_group_decision_weighted_mean():
    agents = [Agent(model=None) for _ in range(2)]
    system = MultiAgentSystem(agents)
    system.step_all = lambda obs, states=None: [np.array([1]), np.array([3])]
    result = system.group_decision([None, None], method="weighted_mean", weights=[0.75, 0.25])
    assert np.allclose(result, [1.5])

def test_group_decision_majority_vote():
    agents = [Agent(model=None) for _ in range(5)]
    system = MultiAgentSystem(agents)
    system.step_all = lambda obs, states=None: [0, 1, 1, 2, 1]
    result = system.group_decision([None]*5, method="majority_vote")
    assert result == 1 or np.all(result == 1)

def test_group_decision_custom():
    agents = [Agent(model=None) for _ in range(3)]
    system = MultiAgentSystem(agents)
    system.step_all = lambda obs, states=None: [np.array([2]), np.array([4]), np.array([6])]
    result = system.group_decision([None]*3, method="custom", custom_fn=lambda acts: np.min(acts, axis=0))
    assert np.allclose(result, [2])

def test_group_decision_law_block():
    clear_laws('group')
    def block_large(x, state=None):
        return np.all(x < 2)
    register_law(block_large, category='group')
    agents = [Agent(model=None) for _ in range(2)]
    system = MultiAgentSystem(agents)
    system.step_all = lambda obs, states=None: [np.array([3]), np.array([3])]
    with pytest.raises(LawViolation):
        system.group_decision([None, None], method="mean")
