import numpy as np
import pytest
from src.core.som_cluster import SOMClusterer
from src.core.multiagent import MultiAgentSystem

class DummyAgent:
    def __init__(self, features):
        self.features = features
        self.group = None

def test_som_clusterer_basic():
    # Create simple feature matrix with two clear clusters
    features = np.array([
        [0.1, 0.2], [0.2, 0.1], [0.15, 0.15],  # Cluster 1
        [0.8, 0.9], [0.9, 0.8], [0.85, 0.85]   # Cluster 2
    ])
    som = SOMClusterer(input_dim=2, som_shape=(2, 2), num_iteration=200)
    som.fit(features)
    clusters = som.predict(features)
    # There should be at least two unique clusters
    assert len(set(clusters)) >= 2

def test_auto_group_by_som():
    # Create agents with clusterable features
    features = np.array([
        [0.0, 0.0], [0.05, 0.05], [0.1, 0.1],   # Group 1
        [1.0, 1.0], [1.05, 1.05], [1.1, 1.1]    # Group 2
    ])
    agents = [DummyAgent(f) for f in features]
    mas = MultiAgentSystem(agents)
    mas.auto_group_by_som([a.features for a in agents], som_shape=(2, 2), num_iteration=200)
    # Collect group assignments
    group_assignments = [a.group for a in agents]
    # There should be at least two unique groups
    assert len(set(group_assignments)) >= 2
    # Agents with similar features should be in the same group
    assert group_assignments[0] == group_assignments[1] == group_assignments[2]
    assert group_assignments[3] == group_assignments[4] == group_assignments[5]
