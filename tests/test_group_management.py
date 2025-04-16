"""
test_group_management.py
Minimal test for dynamic group formation, joining, leaving, and dissolution in MultiAgentSystem.
"""
from src.core.agent import Agent
from src.core.multiagent import MultiAgentSystem

# Create dummy agents
agents = [Agent(model=None) for _ in range(4)]
system = MultiAgentSystem(agents)

# Form a group with agents 0 and 1
system.form_group('A', [0, 1])
assert agents[0].group == 'A'
assert agents[1].group == 'A'
assert 'A' in system.groups and 0 in system.groups['A'] and 1 in system.groups['A']

# Agent 2 joins group A
system.join_group(2, 'A')
assert agents[2].group == 'A'
assert 2 in system.groups['A']

# Agent 1 leaves group A
system.leave_group(1)
assert agents[1].group is None
assert 1 not in system.groups['A']

# Dissolve group A
system.dissolve_group('A')
assert all(agent.group is None for agent in agents)
assert 'A' not in system.groups

print("All group management tests passed.")
