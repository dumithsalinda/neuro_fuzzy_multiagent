import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


def render_agent_positions(
    positions: List[Tuple[float, float]], agents: List[Any]
) -> None:
    """
    Visualize agent positions and group assignments in a dataframe and scatter plot. Optionally highlight leaders if present.
    """

    # Enhanced agent position visualization
    st.write("Agent positions:", positions)
    st.write("Agents:", agents)
    # If positions and groups available, plot scatter by group
    groups = [getattr(agent, "group", None) for agent in agents]
    df = pd.DataFrame(
        {
            "Agent": list(range(len(agents))),
            "X": [p[0] if p is not None else None for p in positions],
            "Y": [p[1] if p is not None else None for p in positions],
            "Group": groups,
        }
    )
    st.dataframe(df)
    # Scatter plot if positions are available
    if df["X"].notnull().all() and df["Y"].notnull().all():
        fig, ax = plt.subplots(figsize=(6, 4))
        group_ids = list(set(groups))
        group_colors = {gid: plt.cm.tab10(i % 10) for i, gid in enumerate(group_ids)}
        for i, agent in enumerate(agents):
            color = group_colors.get(getattr(agent, "group", None), "gray")
            x, y = positions[i] if positions[i] is not None else (0, 0)
            marker = "*" if getattr(agent, "is_leader", False) else "o"
            size = 250 if getattr(agent, "is_leader", False) else 100
            ax.scatter(
                x,
                y,
                c=[color],
                marker=marker,
                s=size,
                edgecolor="black",
                label=f"{getattr(agent, 'group', '')}{' (Leader)' if getattr(agent, 'is_leader', False) else ''}",
            )
            ax.text(x, y + 0.08, f"{i}", ha="center", fontsize=10)
        # Unique legend
        handles = []
        for gid in group_ids:
            is_leader = any(
                getattr(agent, "is_leader", False)
                and getattr(agent, "group", None) == gid
                for agent in agents
            )
            marker = "*" if is_leader else "o"
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="w",
                    label=f"{gid}{' (Leader)' if is_leader else ''}",
                    markerfacecolor=group_colors[gid],
                    markeredgecolor="black",
                    markersize=12 if is_leader else 8,
                )
            )
        ax.legend(handles=handles, loc="best")
        ax.set_title("Agent Positions by Group & Leader")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        st.pyplot(fig)


def render_group_modules(group_modules: dict) -> None:
    """
    Display group module assignments in a dataframe.
    """

    # Enhanced group module visualization
    if group_modules:
        st.write("Group Modules:")
        st.dataframe(pd.DataFrame.from_dict(group_modules, orient="index"))
    else:
        st.write("No group modules available.")


def render_group_knowledge(mas) -> None:
    """
    Display group-level knowledge for each group in the MultiAgentSystem.
    """
    st.markdown("---")
    st.subheader("Group-Level Knowledge/Policy")

    for group_id, members in mas.groups.items():
        st.markdown(f"**Group {group_id}:** Members: {list(members)}")
        # Show leader's or aggregated knowledge
        leader_idx = mas.group_leaders.get(group_id, list(members)[0])
        leader_knowledge = mas.agents[leader_idx].share_knowledge()
        st.write("Leader's Knowledge:")
        st.json(json.dumps(leader_knowledge, default=str, indent=2))


def render_som_grid_with_agents(mas: Any, som: Any) -> None:
    """
    Visualize the SOM grid as a heatmap and show agent mappings (color-coded by group).
    mas: MultiAgentSystem
    som: SOMClusterer instance (must have get_weights() and som_shape)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    st.markdown("---")
    st.subheader("SOM Grid Visualization")
    try:
        weights = som.get_weights() if som else None
        if weights is None:
            st.info("SOM not trained or available.")
            return
        grid_shape = som.som_shape
        grid_norms = np.linalg.norm(weights, axis=-1)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(grid_norms, cmap="Blues", annot=True, fmt=".2f", cbar=True, ax=ax)
        clusters = som.predict(
            [
                getattr(agent, "last_observation", np.zeros(weights.shape[-1]))
                for agent in mas.agents
            ]
        )
        groups = [getattr(agent, "group", None) for agent in mas.agents]
        group_ids = list(set(groups))
        group_colors = {gid: plt.cm.tab10(i % 10) for i, gid in enumerate(group_ids)}
        for idx, (node, group) in enumerate(zip(clusters, groups)):
            ax.scatter(
                node[1] + 0.5,
                node[0] + 0.5,
                color=group_colors.get(group, "gray"),
                s=200,
                label=f"Agent {idx} (G:{group})",
            )
        ax.set_title("SOM Grid with Agent Mappings")
        ax.set_xlabel("SOM X")
        ax.set_ylabel("SOM Y")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to render SOM grid: {e}")


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
        ax2.scatter(
            x,
            y,
            c=[color],
            marker=marker,
            s=size,
            edgecolor="black",
            label=f"{getattr(agent, 'group', '')}{' (Leader)' if getattr(agent, 'is_leader', False) else ''}",
        )
        ax2.text(x, y + 0.08, f"{i}", ha="center", fontsize=10)
    handles = []
    for gid in group_ids:
        is_leader = any(
            getattr(agent, "is_leader", False) and getattr(agent, "group", "") == gid
            for agent in agents
        )
        marker = "*" if is_leader else "o"
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                label=f"{gid}{' (Leader)' if is_leader else ''}",
                markerfacecolor=group_colors[gid],
                markeredgecolor="black",
                markersize=12 if is_leader else 8,
            )
        )
    ax2.legend(handles=handles, loc="best")
    ax2.set_title("Agent Positions by Group & Leader")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True)
    st.pyplot(fig2)
    return group_ids, group_colors


def plot_som_grid(
    group_ids: List[str], agents: List[Any], group_colors: Dict[str, Any]
) -> None:
    """
    Plot a SOM grid showing agent cluster assignments by group.
    """
    som_groups = [g for g in group_ids if isinstance(g, str) and g.startswith("som_")]
    if som_groups:
        som_coords = [tuple(map(int, g.split("_")[1:])) for g in som_groups]
        x_max = max(x for x, y in som_coords) + 1
        y_max = max(y for x, y in som_coords) + 1
        fig_som, ax_som = plt.subplots(figsize=(5, 5))
        ax_som.set_xticks(np.arange(x_max))
        ax_som.set_yticks(np.arange(y_max))
        ax_som.grid(
            True, which="both", color="lightgray", linestyle="--", linewidth=0.7
        )
        for g in som_groups:
            x, y = map(int, g.split("_")[1:])
            members = [
                i for i, agent in enumerate(agents) if getattr(agent, "group", "") == g
            ]
            ax_som.scatter(
                x,
                y,
                s=200,
                c=[group_colors[g]],
                marker="s",
                edgecolor="black",
                label=f"{g} ({len(members)})",
            )
            for i in members:
                ax_som.text(
                    x, y, str(i), color="black", fontsize=10, ha="center", va="center"
                )
        ax_som.set_xlim(-0.5, x_max - 0.5)
        ax_som.set_ylim(-0.5, y_max - 0.5)
        ax_som.set_title("SOM Grid: Agent Cluster Assignments")
        ax_som.set_xlabel("SOM X")
        ax_som.set_ylabel("SOM Y")
        ax_som.legend(loc="best", fontsize=8)
        st.pyplot(fig_som)


def render_agent_reward_and_qtable(agents, reward_history=None):
    """
    Render reward history line chart and Q-table for each agent.
    """
    import streamlit as st

    for i, agent in enumerate(agents):
        st.subheader(f"Agent {i+1} Reward History")
        r = list(reward_history.get(agent, [])) if reward_history else []
        st.line_chart(r if r else [0])
        if hasattr(agent, "q_table"):
            st.text(f"Q-table for Agent {i+1}:")
            st.write(agent.q_table)


def render_group_analytics(mas, reward_history=None) -> None:
    """
    Display analytics for agent groups, optionally using reward history.
    Displays group-level analytics: average reward, group size, cohesion, diversity.
    Args:
        mas: MultiAgentSystem
        reward_history: dict of agent_idx -> deque of recent rewards
    """
    import numpy as np
    import pandas as pd

    st.markdown("---")
    st.subheader("Group-Level Analytics")
    data = []
    for group_id, members in mas.groups.items():
        group_agents = [mas.agents[idx] for idx in members]
        # Average reward
        avg_reward = None
        if reward_history:
            rewards = [
                (
                    np.mean(reward_history.get(idx, []))
                    if len(reward_history.get(idx, [])) > 0
                    else 0.0
                )
                for idx in members
            ]
            avg_reward = np.mean(rewards) if rewards else None
        # Group size
        size = len(members)
        # Cohesion: mean pairwise similarity (using last_observation)
        obs = [
            getattr(agent, "last_observation", None)
            for agent in group_agents
            if getattr(agent, "last_observation", None) is not None
        ]
        cohesion = None
        if len(obs) > 1:
            obs = np.stack(obs)
            dists = np.linalg.norm(obs[:, None, :] - obs[None, :, :], axis=-1)
            mean_dist = np.mean(dists) if dists.size > 0 else 0.0
            cohesion = 1.0 / (1.0 + mean_dist)  # Higher = more cohesive
        # Diversity: variance of observations
        diversity = None
        if len(obs) > 1:
            var = np.var(obs, axis=0)
            diversity = float(np.mean(var)) if var.size > 0 else 0.0
        data.append(
            {
                "Group": group_id,
                "Size": size,
                "AvgReward": avg_reward,
                "Cohesion": cohesion,
                "Diversity": diversity,
            }
        )
    df = pd.DataFrame(data)
    st.dataframe(df)
