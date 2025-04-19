"""
test_multiagent_som.py

Test SOM-based dynamic grouping in MultiAgentSystem.
"""
import numpy as np
from src.core.multiagent import MultiAgentSystem

class DummyAgent:
    def __init__(self, obs):
        self.obs = obs
        self.group = None
    def get_observation(self):
        return self.obs


def test_auto_group_by_som():
    # Create 9 agents with 2D features clustered in 3 groups
    features = np.array([
        [0.1, 0.2], [0.2, 0.15], [0.12, 0.18],   # Cluster 1
        [2.0, 2.1], [2.2, 2.05], [2.1, 2.15],     # Cluster 2
        [4.0, 4.1], [4.2, 4.05], [4.1, 4.15],     # Cluster 3
    ])
    agents = [DummyAgent(f) for f in features]
    mas = MultiAgentSystem(agents)
    feature_matrix = [a.get_observation() for a in agents]
    mas.auto_group_by_som(feature_matrix, som_shape=(3, 1), num_iteration=200)
    # There should be 3 groups (one per SOM node)
    assert len(mas.groups) == 3
    # Each agent should be assigned to a group
    for idx, agent in enumerate(agents):
        found = any(idx in members for members in mas.groups.values())
        assert found, f"Agent {idx} not assigned to any group"
    # Print group assignments for debugging
    for group_id, members in mas.groups.items():
        print(f"Group {group_id}: agents {sorted(members)}")
