"""
test_federated_aggregation.py

Test federated aggregation for Q-table, neural net, and fuzzy rule agents using Ray.
"""

import numpy as np
import pytest
import ray

from src.core.distributed_agent_executor import RayAgentWrapper
from src.core.federated_aggregation import federated_update


class QTableAgent:
    def __init__(self, qtable):
        self.qtable = qtable

    def share_knowledge(self):
        return self.qtable

    def integrate_online_knowledge(self, q):
        self.qtable = q


class NNAgent:
    def __init__(self, weights):
        # Store weights as numpy arrays, preserving shape
        self.weights = {k: np.array(v) for k, v in weights.items()}

    def share_knowledge(self):
        # Return weights as numpy arrays, preserving shape
        return {k: np.array(v) for k, v in self.weights.items()}

    def integrate_online_knowledge(self, w):
        # Store weights as numpy arrays, preserving shape
        self.weights = {k: np.array(v) for k, v in w.items()}


class FuzzyAgent:
    def __init__(self, rules):
        self.rules = rules

    def share_knowledge(self):
        return self.rules

    def integrate_online_knowledge(self, rules):
        self.rules = rules


@pytest.mark.parametrize(
    "AgentClass, knowledge_list, expected_type",
    [
        (
            QTableAgent,
            [
                {("s1", "a1"): 1.0, ("s2", "a2"): 2.0},
                {("s1", "a1"): 3.0, ("s2", "a2"): 4.0},
            ],
            dict,
        ),
        (
            NNAgent,
            [
                {"w1": np.array([1.0, 2.0]), "w2": np.array([3.0])},
                {"w1": np.array([3.0, 4.0]), "w2": np.array([5.0])},
            ],
            dict,
        ),
        (
            FuzzyAgent,
            [
                [{"param": 1.0, "label": "A"}, {"param": 2.0, "label": "B"}],
                [{"param": 3.0, "label": "A"}, {"param": 4.0, "label": "C"}],
            ],
            list,
        ),
    ],
)
def test_federated_update(AgentClass, knowledge_list, expected_type):
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    if AgentClass.__name__ == 'NNAgent':
        print('Test knowledge_list input:')
        for i, k in enumerate(knowledge_list):
            for key, val in k.items():
                print(f'Agent {i}, key {key}: type={type(val)}, shape={np.array(val).shape}, value={val}')
    agents = [AgentClass(knowledge) for knowledge in knowledge_list]
    ray_agents = [RayAgentWrapper.remote(agent) for agent in agents]
    agg = federated_update(ray_agents)
    # After aggregation, all agents should have the same knowledge
    knowledges = ray.get([a.get_knowledge.remote() for a in ray_agents])
    if AgentClass.__name__ == "NNAgent":
        print("NNAgent knowledges:")
        for i, k in enumerate(knowledges):
            for key, val in k.items():
                print(
                    f"Agent {i}, key '{key}': type={type(val)}, shape={np.array(val).shape}, value={val}"
                )
    assert all(isinstance(k, expected_type) for k in knowledges)
    # Check Q-table average (QTableAgent)
    if AgentClass.__name__ == "QTableAgent":
        assert np.isclose(agg[("s1", "a1")], 2.0)
        assert np.isclose(agg[("s2", "a2")], 3.0)
    # Check NN weights average (NNAgent)
    if AgentClass.__name__ == "NNAgent":
        assert np.allclose(agg["w1"], np.array([2.0, 3.0]))
        assert np.allclose(agg["w2"], np.array([4.0]))
    # Check fuzzy rule average/majority (FuzzyAgent)
    if AgentClass.__name__ == "FuzzyAgent":
        assert np.isclose(agg[0]["param"], 2.0)
        assert agg[0]["label"] == "A"
        assert np.isclose(agg[1]["param"], 3.0)
        assert agg[1]["label"] in ["B", "C"]  # majority vote
    ray.shutdown()
    print(f"Federated aggregation successful for {AgentClass.__name__}")
