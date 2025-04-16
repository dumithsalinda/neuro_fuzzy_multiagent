"""
test_leader_election.py
Minimal test for group leader election in MultiAgentSystem.
"""
import numpy as np
from src.core.agent import Agent
from src.core.multiagent import MultiAgentSystem

# Create dummy agents with positions
agents = [Agent(model=None) for _ in range(4)]
positions = [np.array([0,0]), np.array([0.5,0]), np.array([3,3]), np.array([3.2,2.8])]
for agent, pos in zip(agents, positions):
    agent.position = pos

system = MultiAgentSystem(agents)
system.auto_group_by_proximity(distance_threshold=1.0)
system.elect_leaders()

# Check that the lowest index in each group is the leader
for group_id, members in system.groups.items():
    leader = system.group_leaders[group_id]
    assert leader == min(members)
    assert agents[leader].is_leader
    for idx in members:
        if idx != leader:
            assert not agents[idx].is_leader
print("Leader election test passed.")
