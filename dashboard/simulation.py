import streamlit as st
from src.core.multiagent import MultiAgentSystem
from src.core.som_cluster import SOMClusterer
from src.core.distributed_agent_executor import run_agents_distributed
import numpy as np
import streamlit as st

def som_group_agents():
    """
    Cluster agents using SOM based on their current observation vectors.
    Assign group labels and update MultiAgentSystem.
    """
    agents = st.session_state.get("agents", [])
    if not agents or "obs" not in st.session_state:
        st.warning("No agents or observations found for clustering.")
        return
    obs = st.session_state["obs"]
    # Ensure obs is a 2D array (n_agents, n_features)
    obs_matrix = np.array(obs)
    if obs_matrix.ndim == 1:
        obs_matrix = obs_matrix.reshape(-1, 1)
    # Get MultiAgentSystem instance
    mas = st.session_state.get("multiagent_system")
    if mas is None:
        mas = MultiAgentSystem(agents)
        st.session_state["multiagent_system"] = mas
    # Perform SOM-based grouping
    mas.auto_group_by_som(obs_matrix)
    st.session_state["groups"] = mas.groups
    st.success("Agents re-clustered using SOM!")

def simulate_step():
    # Integrated simulation logic with group knowledge sharing and collective actions
    import streamlit as st
    agents = st.session_state.get("agents", [])
    mas = st.session_state.get("multiagent_system")
    obs = st.session_state.get("obs")
    if not agents or mas is None or obs is None:
        st.warning("Simulation not initialized. Make sure agents and observations are available.")
        return
    # Group knowledge sharing if enabled
    if st.session_state.get("enable_group_knowledge", False):
        mas.share_group_knowledge(mode='average')
    # Collective action selection
    collective_mode = st.session_state.get("collective_mode", "individual")
    actions = []
    agent_rules = []
    # --- Distributed agent execution toggle ---
    distributed_mode = st.session_state.get("distributed_execution", False)
    if collective_mode == "individual":
        if distributed_mode:
            # Distributed execution using Ray
            actions = run_agents_distributed(agents, obs)
            agent_rules = [getattr(agent, "last_rule", None) for agent in agents]
        else:
            for agent, o in zip(agents, obs):
                action = agent.act(o)
                actions.append(action)
                # Try to get rule/attention info if available
                rule = getattr(agent, "last_rule", None)
                agent_rules.append(rule)
    else:
        actions = mas.collective_action_selection(obs, mode=collective_mode)
        agent_rules = [None] * len(agents)
# To enable distributed execution, set st.session_state["distributed_execution"] = True in your Streamlit sidebar or controls.
    # Apply actions to environment (example, adapt as needed)
    env = st.session_state.get("env")
    if env is not None:
        next_obs, rewards, done, info = env.step(actions)
        st.session_state["obs"] = next_obs
        st.session_state["done"] = done
        st.session_state["rewards"] = rewards
        for i, agent in enumerate(agents):
            agent.observe(rewards[i])
    st.session_state["step"] = st.session_state.get("step", 0) + 1
    # --- Episode memory recording ---
    if "episode_memory" not in st.session_state:
        st.session_state["episode_memory"] = []
    step_data = []
    for i, agent in enumerate(agents):
        step_data.append({
            "agent": i,
            "obs": obs[i],
            "action": actions[i],
            "reward": st.session_state.get("rewards", [None]*len(agents))[i],
            "group": getattr(agent, "group", None),
            "rule": agent_rules[i],
        })
    st.session_state["episode_memory"].append(step_data)

def run_batch_experiments(n_experiments, agent_counts_list, seeds_list, n_steps, fast_mode=True):
    """
    Run multiple experiments in batch/parallel with varying agent counts and seeds.
    Returns a list of summary dicts for each experiment.
    Calculates advanced metrics: diversity, group_stability, intervention_count.
    If fast_mode is True, skips unnecessary UI/plotting for higher speed.
    """
    import numpy as np
    from src.core.agent import Agent
    from src.core.multiagent import MultiAgentSystem
    import random
    results = []
    exp_id = 0
    for agent_count in agent_counts_list:
        for seed in seeds_list:
            exp_id += 1
            np.random.seed(seed)
            random.seed(seed)
            # Create agents and MAS
            agents = [Agent(model=None) for _ in range(agent_count)]
            mas = MultiAgentSystem(agents)
            # Dummy env: each step returns random reward per agent
            rewards_history = []
            group_history = []
            intervention_count = 0
            obs = [np.zeros(2) for _ in range(agent_count)]
            for step in range(n_steps):
                # (Replace with real env/logic as needed)
                actions = [0 for _ in agents]
                rewards = np.random.uniform(0, 1, size=agent_count)
                rewards_history.append(np.mean(rewards))
                # Simulate agent observation update
                for i, agent in enumerate(agents):
                    agent.last_observation = obs[i]
                    agent.observe(rewards[i])
                # Simulate group assignment and interventions
                group_labels = np.random.randint(0, max(1, agent_count // 10), size=agent_count)
                group_history.append(group_labels)
                if np.random.rand() < 0.05:
                    intervention_count += 1
            # Diversity: mean number of unique groups per step
            diversity = float(np.mean([len(set(g)) for g in group_history])) if group_history else 0.0
            # Group stability: mean fraction of agents that stayed in the same group as previous step
            group_stability = 0.0
            if len(group_history) > 1:
                stabilities = []
                for prev, curr in zip(group_history[:-1], group_history[1:]):
                    stabilities.append(np.mean(np.array(prev) == np.array(curr)))
                group_stability = float(np.mean(stabilities))
            result = {
                "experiment": exp_id,
                "agent_count": agent_count,
                "seed": seed,
                "mean_reward": float(np.mean(rewards_history)) if len(rewards_history) > 0 else 0.0,
                "max_reward": float(np.max(rewards_history)) if len(rewards_history) > 0 else 0.0,
                "min_reward": float(np.min(rewards_history)) if len(rewards_history) > 0 else 0.0,
                "std_reward": float(np.std(rewards_history)) if len(rewards_history) > 0 else 0.0,
                "diversity": diversity,
                "group_stability": group_stability,
                "intervention_count": intervention_count,
            }
            results.append(result)
    return results
