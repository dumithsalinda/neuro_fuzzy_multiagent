import numpy as np

def export_agent_group_state(mas):
    """
    Utility to extract agent positions, group assignments, and group modules from a MultiAgentSystem.
    Returns:
        - positions: np.ndarray of shape (n_agents, 2) or None
        - group_assignments: list of group ids
        - group_modules: dict mapping group_id to module info
    """
    positions = []
    group_assignments = []
    for agent in mas.agents:
        pos = getattr(agent, 'position', None)
        if pos is not None:
            positions.append(np.array(pos))
        else:
            positions.append(None)
        group_assignments.append(getattr(agent, 'group', None))
    if all(p is not None for p in positions):
        positions = np.stack(positions)
    else:
        positions = None
    return {
        'positions': positions,
        'group_assignments': group_assignments,
        'group_modules': mas.group_modules.copy()
    }
