"""
test_auto_group.py
Minimal test for auto_group_by_proximity in MultiAgentSystem.
"""

import numpy as np

from neuro_fuzzy_multiagent.core.agents.agent import Agent
from neuro_fuzzy_multiagent.core.management.multiagent import MultiAgentSystem

# Create dummy agents with positions
agents = [Agent(model=None) for _ in range(4)]
positions = [
    np.array([0, 0]),
    np.array([0.5, 0]),
    np.array([3, 3]),
    np.array([3.2, 2.8]),
]
for agent, pos in zip(agents, positions):
    agent.position = pos

system = MultiAgentSystem(agents)
system.auto_group_by_proximity(distance_threshold=1.0)

# Expect two groups: agents 0/1 and agents 2/3
group_ids = [agent.group for agent in agents]
assert (
    group_ids[0] == group_ids[1]
    and group_ids[2] == group_ids[3]
    and group_ids[0] != group_ids[2]
)
assert len(system.groups) == 2
print("Auto-grouping by proximity test passed.")
