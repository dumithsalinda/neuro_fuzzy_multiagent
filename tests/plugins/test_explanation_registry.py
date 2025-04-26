import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)
import pytest
from src.core.plugins.explanation_registry import (
    register_explanation,
    get_explanation,
    clear_explanations,
)


class DummyAgent:
    def explain_action(self, obs=None):
        return "default explanation"


def custom_explainer(agent, obs=None):
    return "custom explanation for {}".format(obs)


def test_register_and_get_custom_explanation():
    clear_explanations()
    # By default, falls back to agent's own method
    agent = DummyAgent()
    assert get_explanation(agent, obs=1) == "default explanation"
    # Register a custom explanation function
    register_explanation(DummyAgent)(custom_explainer)
    assert get_explanation(agent, obs=2) == "custom explanation for 2"


def test_not_implemented():
    class NoExplainAgent:
        pass

    clear_explanations()
    agent = NoExplainAgent()
    with pytest.raises(NotImplementedError):
        get_explanation(agent, obs=3)
