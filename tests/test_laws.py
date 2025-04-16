import pytest
import numpy as np
from src.laws import register_law, remove_law, clear_laws, enforce_laws, LAWS
from src.core.agent import Agent

# --- Utility law functions ---
def law_positive(action, state):
    return np.all(action >= 0)
def law_below_five(action, state):
    return np.all(action < 5)

def test_register_and_enforce_government_law():
    clear_laws()
    register_law('government', law_positive)
    enforce_laws(np.array([1, 2]), state=None, categories=['government'])
    with pytest.raises(Exception):
        enforce_laws(np.array([-1, 2]), state=None, categories=['government'])


def test_register_and_enforce_organization_law():
    clear_laws()
    register_law('organization', law_below_five)
    enforce_laws(np.array([1, 2]), state=None, categories=['organization'])
    with pytest.raises(Exception):
        enforce_laws(np.array([6]), state=None, categories=['organization'])


def test_enforce_multiple_categories():
    clear_laws()
    register_law('government', law_positive)
    register_law('organization', law_below_five)
    enforce_laws(np.array([1, 2]), state=None, categories=['government', 'organization'])
    with pytest.raises(Exception):
        enforce_laws(np.array([-1, 2]), state=None, categories=['government', 'organization'])
    with pytest.raises(Exception):
        enforce_laws(np.array([6]), state=None, categories=['government', 'organization'])


def test_agent_act_enforces_laws(monkeypatch):
    clear_laws()
    register_law('personal', law_positive)
    # Patch enforce_laws to always check 'personal'
    monkeypatch.setattr('src.core.agent.enforce_laws', lambda a, s: enforce_laws(a, s, categories=['personal']))
    agent = Agent(model=None, policy=lambda obs, model: np.array([-1]))
    with pytest.raises(Exception):
        agent.act(np.array([0]))
    # Now allow positive actions
    agent = Agent(model=None, policy=lambda obs, model: np.array([1]))
    agent.act(np.array([0]))  # Should not raise
