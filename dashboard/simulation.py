from typing import Any, List

import numpy as np
import streamlit as st

from neuro_fuzzy_multiagent.core.management.distributed_agent_executor import (
    run_agents_distributed,
)
from neuro_fuzzy_multiagent.core.management.multiagent import MultiAgentSystem
from neuro_fuzzy_multiagent.core.neural_networks.som_cluster import SOMClusterer


def som_group_agents() -> None:
    """
    Cluster agents using SOM based on their current observation vectors.
    Assign group labels and update MultiAgentSystem.
    Provides user feedback on errors or missing data.
    """
    agents: List[Any] = st.session_state.get("agents", [])
    if not agents or "obs" not in st.session_state:
        st.warning("No agents or observations found for clustering.")
        return
    obs = st.session_state["obs"]
    obs_matrix = np.array(obs)
    if obs_matrix.ndim == 1:
        obs_matrix = obs_matrix.reshape(-1, 1)
    mas = st.session_state.get("multiagent_system")
    if mas is None:
        mas = MultiAgentSystem(agents)
        st.session_state["multiagent_system"] = mas
    try:
        mas.auto_group_by_som(obs_matrix)
        st.session_state["groups"] = mas.groups
        st.success("Agents re-clustered using SOM!")
    except Exception as e:
        st.error(f"SOM clustering failed: {e}")


def simulate_step() -> None:
    """
    Run a simulation step with adversarial perturbation, agent action selection, feedback, and online learning.
    Updates session state with new observations, rewards, and step count.
    Provides user feedback and robust error handling for all critical operations.
    """
    """
    Run a simulation step with adversarial perturbation, agent action selection, feedback, and online learning.
    Updates session state with new observations, rewards, and step count.
    """
    import random

    import numpy as np
    import requests

    from neuro_fuzzy_multiagent.core.agents.agent import Agent
    from neuro_fuzzy_multiagent.core.agents.multimodal_fusion_agent import (
        MultiModalFusionAgent,
    )

    # --- Get session state ---
    try:
        env = st.session_state.get("env", None)
        agents = st.session_state.get("agents", [])
        feedback = st.session_state.get("feedback", {})
        online_learning_enabled = st.session_state.get("online_learning_enabled", True)
        adversarial_enabled = st.session_state.get("adversarial_enabled", False)
        adversarial_agents = st.session_state.get("adversarial_agents", [])
        adversarial_type = st.session_state.get("adversarial_type", "None")
        adversarial_strength = st.session_state.get("adversarial_strength", 0.1)
        orig_obs = list(st.session_state.get("obs", []))
        if env is None or not agents or not orig_obs:
            st.error(
                "Simulation state initialization failed: missing required session state."
            )
            return
    except Exception as e:
        st.error(f"Simulation state initialization failed: {e}")
        return
    perturbed_obs = []

    def to_scalar_action(a):
        if isinstance(a, np.ndarray):
            return int(a.item()) if a.size == 1 else int(a.flat[0])
        return int(a) if isinstance(a, (np.integer, np.floating)) else a

    for i, obs in enumerate(orig_obs):
        is_dqn = getattr(agents[i], "__class__", type(agents[i])).__name__ == "DQNAgent"
        if (
            adversarial_enabled
            and i in adversarial_agents
            and adversarial_type != "None"
        ):
            obs_arr = np.array(obs, dtype=np.float32)
            if adversarial_type == "Gaussian Noise":
                obs_arr = obs_arr + np.random.normal(
                    0, adversarial_strength, size=obs_arr.shape
                )
            elif adversarial_type == "Uniform Noise":
                obs_arr = obs_arr + np.random.uniform(
                    -adversarial_strength, adversarial_strength, size=obs_arr.shape
                )
            elif adversarial_type == "Zeros":
                obs_arr = np.zeros_like(obs_arr)
            elif adversarial_type == "Max Value":
                obs_arr = np.ones_like(obs_arr) * adversarial_strength
            elif adversarial_type == "FGSM (Targeted, DQN)" and is_dqn:
                try:
                    import torch

                    obs_tensor = (
                        torch.tensor(obs_arr, requires_grad=True).unsqueeze(0).float()
                    )
                    qvals = agents[i].model(obs_tensor)
                    action = torch.argmax(qvals, dim=1)
                    loss = -qvals[0, action]
                    loss.backward()
                    grad_sign = obs_tensor.grad.data.sign().squeeze(0).numpy()
                    obs_arr = obs_arr + adversarial_strength * grad_sign
                except Exception:
                    obs_arr = obs_arr + np.random.normal(
                        0, adversarial_strength, size=obs_arr.shape
                    )
            perturbed_obs.append(obs_arr.tolist())
        else:
            perturbed_obs.append(obs)
    st.session_state.perturbed_obs = perturbed_obs
    actions = []
    for i, (agent, obs) in enumerate(zip(agents, perturbed_obs)):
        if isinstance(agent, MultiModalFusionAgent) and not (
            isinstance(obs, list) and len(obs) == 2
        ):
            img_dim, txt_dim = agent.model.input_dims
            obs = [np.random.randn(img_dim), np.random.randn(txt_dim)]
        fb = feedback.get(
            i, {"approve": "Approve", "override_action": None, "custom_reward": None}
        )
        if fb["approve"] == "Reject":
            action = getattr(agent, "last_action", 0)
        elif fb["approve"] == "Override" and fb["override_action"] is not None:
            try:
                action = (
                    type(obs[0])(fb["override_action"])
                    if hasattr(obs, "__getitem__")
                    else int(fb["override_action"])
                )
            except Exception:
                action = fb["override_action"]
        else:
            action = to_scalar_action(agent.act(obs))
        actions.append(action)
    next_obs, rewards, done = env.step(actions)
    # Apply custom rewards if provided
    for i, fb in feedback.items():
        if fb.get("custom_reward") is not None:
            try:
                rewards[i] = float(fb["custom_reward"])
            except Exception:
                pass
    for i, agent in enumerate(st.session_state.agents):
        agent.integrate_online_knowledge({"step": st.session_state.step})
        agent.online_knowledge = {"step": st.session_state.step}
        others = [a for a in st.session_state.agents if a is not agent]
        if others:
            recipient = random.choice(others)
            msg = {
                "from": i,
                "step": st.session_state.step,
                "knowledge": agent.online_knowledge,
            }
            agent.send_message(msg, recipient)
        try:
            agent.observe(rewards[i], next_obs[i], done)
        except Exception as e:
            if e.__class__.__name__ == "LawViolation":
                if not hasattr(agent, "law_violations"):
                    agent.law_violations = 0
                agent.law_violations += 1
            else:
                raise
        if online_learning_enabled:
            try:
                api_url = "http://localhost:8000/learn/online"
                headers = {"X-API-Key": "mysecretkey"}
                obs_str = ",".join(str(x) for x in st.session_state.obs[i])
                next_obs_str = ",".join(str(x) for x in next_obs[i])
                target_str = f"{rewards[i]},{next_obs_str},{int(done)}"
                data = {"agent_id": i, "input": obs_str, "target": target_str}
                requests.post(api_url, data=data, headers=headers, timeout=2)
            except Exception as ex:
                st.session_state["online_learning_log"] = (
                    f"Online update failed for agent {i}: {ex}"
                )
    st.session_state.obs = next_obs
    st.session_state.rewards = rewards
    st.session_state.done = done
    st.session_state.step += 1

    """
    Run a simulation step with group knowledge sharing and collective actions.
    Updates session state with new observations, rewards, and step count.
    """

    # Integrated simulation logic with group knowledge sharing and collective actions

    agents = st.session_state.get("agents", [])
    mas = st.session_state.get("multiagent_system")
    obs = st.session_state.get("obs")
    if not agents or mas is None or obs is None:
        st.warning(
            "Simulation not initialized. Make sure agents and observations are available."
        )
        return
    # Group knowledge sharing if enabled
    if st.session_state.get("enable_group_knowledge", False):
        mas.share_group_knowledge(mode="average")
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
        step_data.append(
            {
                "agent": i,
                "obs": obs[i],
                "action": actions[i],
                "reward": st.session_state.get("rewards", [None] * len(agents))[i],
                "group": getattr(agent, "group", None),
                "rule": agent_rules[i],
            }
        )
    if "episode_memory" not in st.session_state:
        st.session_state["episode_memory"] = []
    st.session_state["episode_memory"].append(step_data)


def run_batch_experiments(
    n_experiments: int,
    agent_counts_list: list,
    seeds_list: list,
    n_steps: int,
    fast_mode: bool = True,
) -> list:
    """
    Run multiple experiments in batch/parallel with varying agent counts and seeds.
    Returns a list of summary dicts for each experiment.
    Calculates advanced metrics: diversity, group_stability, intervention_count.
    If fast_mode is True, skips unnecessary UI/plotting for higher speed.
    """

    """
    Run multiple experiments in batch/parallel with varying agent counts and seeds.
    Returns a list of summary dicts for each experiment.
    Calculates advanced metrics: diversity, group_stability, intervention_count.
    If fast_mode is True, skips unnecessary UI/plotting for higher speed.
    """
    import random

    import numpy as np

    from neuro_fuzzy_multiagent.core.agents.agent import Agent
    from neuro_fuzzy_multiagent.core.management.multiagent import MultiAgentSystem

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
                group_labels = np.random.randint(
                    0, max(1, agent_count // 10), size=agent_count
                )
                group_history.append(group_labels)
                if np.random.rand() < 0.05:
                    intervention_count += 1
            # Diversity: mean number of unique groups per step
            diversity = (
                float(np.mean([len(set(g)) for g in group_history]))
                if group_history
                else 0.0
            )
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
                "mean_reward": (
                    float(np.mean(rewards_history)) if len(rewards_history) > 0 else 0.0
                ),
                "max_reward": (
                    float(np.max(rewards_history)) if len(rewards_history) > 0 else 0.0
                ),
                "min_reward": (
                    float(np.min(rewards_history)) if len(rewards_history) > 0 else 0.0
                ),
                "std_reward": (
                    float(np.std(rewards_history)) if len(rewards_history) > 0 else 0.0
                ),
                "diversity": diversity,
                "group_stability": group_stability,
                "intervention_count": intervention_count,
            }
            results.append(result)
    return results
