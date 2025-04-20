import pytest
import numpy as np
from src.core.agents.laws import register_law, remove_law, clear_laws, enforce_laws, list_laws
from src.core.agents.agent import Agent

# --- Utility law functions ---
def law_positive(action, state):
    return np.all(action >= 0)
def law_below_five(action, state):
    return np.all(action < 5)

def test_register_and_enforce_government_law():
    clear_laws()
    register_law('government', law_positive)
    enforce_laws(np.array([1, 2]), state=None, category='government')
    with pytest.raises(Exception):
        enforce_laws(np.array([-1, 2]), state=None, categories=['government'])


def test_register_and_enforce_organization_law():
    clear_laws()
    register_law('organization', law_below_five)
    enforce_laws(np.array([1, 2]), state=None, category='organization')
    with pytest.raises(Exception):
        enforce_laws(np.array([6]), state=None, categories=['organization'])


def test_enforce_multiple_categories():
    clear_laws()
    register_law('government', law_positive)
    register_law('organization', law_below_five)
    enforce_laws(np.array([1, 2]), state=None, category='government')
    enforce_laws(np.array([1, 2]), state=None, category='organization')
    with pytest.raises(Exception):
        enforce_laws(np.array([-1, 2]), state=None, categories=['government', 'organization'])
    with pytest.raises(Exception):
        enforce_laws(np.array([6]), state=None, categories=['government', 'organization'])


def test_agent_act_enforces_laws(monkeypatch):
    clear_laws()
    register_law(law_positive)  # Register in default 'action' category
    agent = Agent(model=None, policy=lambda obs, model: np.array([-1]))
    from src.core.agents.laws import LawViolation
    with pytest.raises(LawViolation):
        agent.act(np.array([0]))
    # Now allow positive actions
    agent = Agent(model=None, policy=lambda obs, model: np.array([1]))
    agent.act(np.array([0]))  # Should not raise
