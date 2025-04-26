import numpy as np
from src.core.agents.agent import NeuroFuzzyAgent


def test_self_organize_changes_rules():
    nn_config = {"input_dim": 1, "hidden_dim": 2, "output_dim": 1}
    fis_config = {
        "X": np.array([[0.0], [1.0]]),
        "y": np.array([0.0, 1.0]),
        "fuzzy_sets_per_input": [
            [
                # Two fuzzy sets for input 0
                type(
                    "FS",
                    (),
                    {
                        "membership": lambda self, x: np.exp(
                            -0.5 * ((x - 0) / 0.5) ** 2
                        ),
                        "params": [0, 0.5],
                        "tune": lambda self, d: None,
                    },
                )(),
                type(
                    "FS",
                    (),
                    {
                        "membership": lambda self, x: np.exp(
                            -0.5 * ((x - 1) / 0.5) ** 2
                        ),
                        "params": [1, 0.5],
                        "tune": lambda self, d: None,
                    },
                )(),
            ]
        ],
    }
    agent = NeuroFuzzyAgent(nn_config, fis_config)
    before = len(agent.model.fis.rules)
    agent.self_organize()
    after = len(agent.model.fis.rules)
    # Should change rule count or structure
    assert before != after or any(
        r1.consequent != r2.consequent
        for r1, r2 in zip(agent.model.fis.rules, agent.model.fis.rules)
    )


def test_self_organize_law_compliance():
    # Law: all consequents must be in [0, 1]
    def law_consequent_in_bounds(action, state):
        return np.all((action >= 0) & (action <= 1))

    nn_config = {"input_dim": 1, "hidden_dim": 2, "output_dim": 1}
    fis_config = {
        "X": np.array([[0.0], [1.0]]),
        "y": np.array([0.0, 1.0]),
        "fuzzy_sets_per_input": [
            [
                type(
                    "FS",
                    (),
                    {
                        "membership": lambda self, x: np.exp(
                            -0.5 * ((x - 0) / 0.5) ** 2
                        ),
                        "params": [0, 0.5],
                        "tune": lambda self, d: None,
                    },
                )(),
                type(
                    "FS",
                    (),
                    {
                        "membership": lambda self, x: np.exp(
                            -0.5 * ((x - 1) / 0.5) ** 2
                        ),
                        "params": [1, 0.5],
                        "tune": lambda self, d: None,
                    },
                )(),
            ]
        ],
    }
    agent = NeuroFuzzyAgent(nn_config, fis_config)
    # Simulate law enforcement after self-organization
    agent.self_organize()
    for rule in agent.model.fis.rules:
        assert 0 <= rule.consequent <= 1
