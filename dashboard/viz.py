import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple

def plot_group_leader_spatial(agents: List[Any]) -> Tuple[list, dict]:
    """
    Plot agent positions by group and leader status. Returns group IDs and color mapping.
    """
    group_ids = list(set(getattr(agent, "group", "") for agent in agents))
    group_colors = {gid: plt.cm.tab10(i % 10) for i, gid in enumerate(group_ids)}
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    for i, agent in enumerate(agents):
        color = group_colors.get(getattr(agent, "group", ""), "gray")
        x, y = getattr(agent, "position", (0, 0))
        marker = "*" if getattr(agent, "is_leader", False) else "o"
        size = 250 if getattr(agent, "is_leader", False) else 100
        ax2.scatter(x, y, c=[color], marker=marker, s=size, edgecolor="black", label=f"{getattr(agent, 'group', '')}{' (Leader)' if getattr(agent, 'is_leader', False) else ''}")
        ax2.text(x, y + 0.08, f"{i}", ha="center", fontsize=10)
    # Unique legend
    handles = []
    for gid in group_ids:
        is_leader = any(getattr(agent, "is_leader", False) and getattr(agent, "group", "") == gid for agent in agents)
        marker = "*" if is_leader else "o"
        handles.append(
            plt.Line2D([0], [0], marker=marker, color="w", label=f"{gid}{' (Leader)' if is_leader else ''}",
                       markerfacecolor=group_colors[gid], markeredgecolor="black", markersize=12 if is_leader else 8)
        )
    ax2.legend(handles=handles, loc="best")
    ax2.set_title("Agent Positions by Group & Leader")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True)
    st.pyplot(fig2)
    return group_ids, group_colors

def plot_som_grid(group_ids: List[str], agents: List[Any], group_colors: Dict[str, Any]) -> None:
    """
    Plot a SOM grid showing agent cluster assignments by group.
    """
    som_groups = [g for g in group_ids if isinstance(g, str) and g.startswith('som_')]
    if som_groups:
        som_coords = [tuple(map(int, g.split('_')[1:])) for g in som_groups]
        x_max = max(x for x, y in som_coords) + 1
        y_max = max(y for x, y in som_coords) + 1
        fig_som, ax_som = plt.subplots(figsize=(5, 5))
        ax_som.set_xticks(np.arange(x_max))
        ax_som.set_yticks(np.arange(y_max))
        ax_som.grid(True, which='both', color='lightgray', linestyle='--', linewidth=0.7)
        for g in som_groups:
            x, y = map(int, g.split('_')[1:])
            members = [i for i, agent in enumerate(agents) if getattr(agent, "group", "") == g]
            ax_som.scatter(x, y, s=200, c=[group_colors[g]], marker='s', edgecolor='black', label=f'{g} ({len(members)})')
            for i in members:
                ax_som.text(x, y, str(i), color='black', fontsize=10, ha='center', va='center')
        ax_som.set_xlim(-0.5, x_max - 0.5)
        ax_som.set_ylim(-0.5, y_max - 0.5)
        ax_som.set_title('SOM Grid: Agent Cluster Assignments')
        ax_som.set_xlabel('SOM X')
        ax_som.set_ylabel('SOM Y')
        ax_som.legend(loc='best', fontsize=8)
        st.pyplot(fig_som)
