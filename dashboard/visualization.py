import streamlit as st

def render_agent_positions(positions, agents):
    # Enhanced agent position visualization
    import pandas as pd
    st.write("Agent positions:", positions)
    st.write("Agents:", agents)
    # If positions and groups available, plot scatter by group
    groups = [getattr(agent, 'group', None) for agent in agents]
    df = pd.DataFrame({
        'Agent': list(range(len(agents))),
        'X': [p[0] if p is not None else None for p in positions],
        'Y': [p[1] if p is not None else None for p in positions],
        'Group': groups
    })
    st.dataframe(df)
    # Scatter plot if positions are available
    if df['X'].notnull().all() and df['Y'].notnull().all():
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x='X', y='Y', hue='Group', data=df, palette='tab10', s=100)
        plt.title('Agent Positions by Group')
        st.pyplot(plt)

def render_group_modules(group_modules):
    # Enhanced group module visualization
    import pandas as pd
    if group_modules:
        st.write("Group Modules:")
        st.dataframe(pd.DataFrame.from_dict(group_modules, orient='index'))
    else:
        st.write("No group modules available.")

def render_group_knowledge(mas):
    """
    Display group-level knowledge for each group in the MultiAgentSystem.
    """
    import json
    st.markdown("---")
    st.subheader("Group-Level Knowledge/Policy")
    for group_id, members in mas.groups.items():
        st.markdown(f"**Group {group_id}:** Members: {list(members)}")
        # Show leader's or aggregated knowledge
        leader_idx = mas.group_leaders.get(group_id, list(members)[0])
        leader_knowledge = mas.agents[leader_idx].share_knowledge()
        st.write("Leader's Knowledge:")
        st.json(json.dumps(leader_knowledge, default=str, indent=2))

def render_som_grid_with_agents(mas, som):
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
    weights = som.get_weights() if som else None
    if weights is None:
        st.info("SOM not trained or available.")
        return
    # SOM grid heatmap (average weight norm per node)
    grid_shape = som.som_shape
    grid_norms = np.linalg.norm(weights, axis=-1)
    plt.figure(figsize=(6, 6))
    sns.heatmap(grid_norms, cmap="Blues", annot=True, fmt=".2f", cbar=True)
    # Agent markers on SOM grid
    clusters = som.predict([getattr(agent, 'last_observation', np.zeros(weights.shape[-1])) for agent in mas.agents])
    groups = [getattr(agent, 'group', None) for agent in mas.agents]
    palette = sns.color_palette("tab10", len(set(groups)))
    for idx, (node, group) in enumerate(zip(clusters, groups)):
        plt.scatter(node[1]+0.5, node[0]+0.5, color=palette[hash(group)%len(palette)], s=200, label=f"Agent {idx} (G:{group})")
    plt.title("SOM Grid with Agent Mappings")
    plt.xlabel("SOM X")
    plt.ylabel("SOM Y")
    st.pyplot(plt)

def render_group_analytics(mas, reward_history=None):
    """
    Display group-level analytics: average reward, group size, cohesion, diversity.
    reward_history: dict of agent_idx -> deque of recent rewards
    """
    import pandas as pd
    import numpy as np
    st.markdown("---")
    st.subheader("Group-Level Analytics")
    data = []
    for group_id, members in mas.groups.items():
        group_agents = [mas.agents[idx] for idx in members]
        # Average reward
        avg_reward = None
        if reward_history:
            rewards = [np.mean(reward_history.get(idx, [])) if len(reward_history.get(idx, [])) > 0 else 0.0 for idx in members]
            avg_reward = np.mean(rewards) if rewards else None
        # Group size
        size = len(members)
        # Cohesion: mean pairwise similarity (using last_observation)
        obs = [getattr(agent, 'last_observation', None) for agent in group_agents if getattr(agent, 'last_observation', None) is not None]
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
        data.append({
            'Group': group_id,
            'Size': size,
            'AvgReward': avg_reward,
            'Cohesion': cohesion,
            'Diversity': diversity
        })
    df = pd.DataFrame(data)
    st.dataframe(df)
