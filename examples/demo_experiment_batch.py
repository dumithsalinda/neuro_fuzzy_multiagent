"""
Demo: Batch experiment runner using ScenarioGenerator and ExperimentManager
Runs real agents in real environments for each scenario.
"""

import os

import numpy as np

from neuro_fuzzy_multiagent.core.experiment_manager import ExperimentManager
from neuro_fuzzy_multiagent.core.management.agent_factory import (
    create_agent_from_config,
)
from neuro_fuzzy_multiagent.core.management.agent_manager import AgentManager
from neuro_fuzzy_multiagent.core.scenario_generator import ScenarioGenerator
from neuro_fuzzy_multiagent.env.environment_factory import EnvironmentFactory
from neuro_fuzzy_multiagent.utils.human_in_the_loop import human_in_the_loop_control
from neuro_fuzzy_multiagent.utils.visualization import (
    plot_agent_explanations,
    plot_rewards,
    plot_rule_activations,
)

# Map scenario env names to factory keys
ENV_MAP = {
    "Gridworld": "multiagent_gridworld",
    "IoTSensorFusionEnv": "iot_sensor_fusion",
}
# Map agent types to example config files
AGENT_CFG_MAP = {
    "DQNAgent": "examples/agent_config_dqn.yaml",
    "NeuroFuzzyAgent": "examples/agent_config_example.yaml",
}

param_grid = {
    "agent_type": ["DQNAgent", "NeuroFuzzyAgent"],
    "agent_count": [2, 4],
    "env": ["Gridworld", "IoTSensorFusionEnv"],
    "seed": [42, 123],
}
scenarios = ScenarioGenerator(param_grid).grid_search()
mgr = ExperimentManager(log_dir="experiments")

for scenario in scenarios:
    np.random.seed(scenario["seed"])
    run = mgr.start_run(scenario)
    # Create environment
    env_key = ENV_MAP[scenario["env"]]
    env = EnvironmentFactory.create(env_key, n_agents=scenario["agent_count"])
    # Create agents using AgentManager for plug-and-play
    agent_mgr = AgentManager()
    agent_cfg_path = AGENT_CFG_MAP[scenario["agent_type"]]
    agent_cfg = __import__("yaml").safe_load(open(agent_cfg_path))
    for _ in range(scenario["agent_count"]):
        agent_mgr.add_agent(agent_cfg)
    agents = agent_mgr.get_agents()
    # Run one episode with random actions
    obs = env.reset()
    total_rewards = [0 for _ in range(scenario["agent_count"])]
    done = False
    steps = 0
    while not done and steps < 50:
        # Human-in-the-loop: every 10 steps, prompt user for control
        if steps > 0 and steps % 10 == 0:
            print("[Plug-and-Play] To swap the first agent, type 'swap' at the prompt.")
            hitl_cmd = human_in_the_loop_control(steps, agent=agents[0], env=env)
            if hitl_cmd == "stop":
                print("Episode stopped by user at step {}.".format(steps))
                break
            elif hitl_cmd.strip().lower() == "swap":
                # Swap the first agent with the other type
                old_agent = agents[0]
                current_type = scenario["agent_type"]
                new_type = (
                    "NeuroFuzzyAgent" if current_type == "DQNAgent" else "DQNAgent"
                )
                new_cfg_path = AGENT_CFG_MAP[new_type]
                new_cfg = __import__("yaml").safe_load(open(new_cfg_path))
                new_agent = agent_mgr.replace_agent(old_agent, new_cfg)
                agents = agent_mgr.get_agents()
                print("[Plug-and-Play] Swapped first agent to {}.".format(new_type))
        # Try to get per-agent observations if available, else share obs
        per_agent_obs = None
        if isinstance(obs, dict) and "agent_positions" in obs:
            # IoTSensorFusionEnv returns dict with agent_positions
            per_agent_obs = [obs for _ in range(scenario["agent_count"])]
        elif (
            isinstance(obs, np.ndarray) and obs.shape[0] >= scenario["agent_count"] * 2
        ):
            # Gridworld returns flat array, assume shared obs
            per_agent_obs = [obs for _ in range(scenario["agent_count"])]
        else:
            per_agent_obs = [obs for _ in range(scenario["agent_count"])]

        actions = []
        for i, agent in enumerate(agents):
            agent_obs = per_agent_obs[i]
            # Use agent's policy if available
            if hasattr(agent, "select_action"):
                try:
                    action = agent.select_action(agent_obs)
                except Exception:
                    action = (
                        np.random.randint(env.action_space)
                        if hasattr(env, "action_space")
                        else 0
                    )
            elif hasattr(agent, "act"):
                try:
                    action = agent.act(agent_obs)
                except Exception:
                    action = (
                        np.random.randint(env.action_space)
                        if hasattr(env, "action_space")
                        else 0
                    )
            else:
                action = (
                    np.random.randint(env.action_space)
                    if hasattr(env, "action_space")
                    else 0
                )
            actions.append(action)

        step_result = env.step(actions)
        # Handle different env return signatures
        if isinstance(step_result, tuple) and len(step_result) == 4:
            obs, rewards, done, info = step_result
            if isinstance(rewards, list):
                for i, r in enumerate(rewards):
                    total_rewards[i] += r
            else:
                for i in range(scenario["agent_count"]):
                    total_rewards[i] += rewards
        else:
            break
        steps += 1
    mean_reward = float(np.mean(total_rewards))
    std_reward = float(np.std(total_rewards))
    result = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "agent_count": scenario["agent_count"],
    }
    mgr.log_results(run, result)
    print(
        "Finished run {} | {} | mean_reward={:.2f}".format(
            run["run_id"], scenario, mean_reward
        )
    )

means = mgr.aggregate_results("mean_reward")
print("All mean rewards: {}".format(means))

# Visualization: plot reward history for all runs
plot_rewards([r for r in means if r is not None], title="Mean Rewards Across Scenarios")

# Visualization: collect and plot agent explanations for the first scenario
# (Assume Q-values or rule activations are returned by explain_action; fallback to zeros if not available)
first_scenario = scenarios[0]
np.random.seed(first_scenario["seed"])
env_key = ENV_MAP[first_scenario["env"]]
env = EnvironmentFactory.create(env_key, n_agents=first_scenario["agent_count"])
agent_cfg_path = AGENT_CFG_MAP[first_scenario["agent_type"]]
agents = [
    create_agent_from_config(__import__("yaml").safe_load(open(agent_cfg_path)))
    for _ in range(first_scenario["agent_count"])
]
obs = env.reset()
explanation_history = [[] for _ in range(first_scenario["agent_count"])]
done = False
steps = 0
while not done and steps < 20:
    per_agent_obs = [obs for _ in range(first_scenario["agent_count"])]
    actions = []
    for i, agent in enumerate(agents):
        agent_obs = per_agent_obs[i]
        if hasattr(agent, "explain_action"):
            try:
                explanation = agent.explain_action(agent_obs)
                # Use Q-values if present, else rule activations, else zeros
                if "q_values" in explanation:
                    explanation_history[i].append(np.max(explanation["q_values"]))
                elif "rule_activations" in explanation:
                    explanation_history[i].append(
                        np.max(explanation["rule_activations"])
                    )
                else:
                    explanation_history[i].append(0)
            except Exception:
                explanation_history[i].append(0)
        else:
            explanation_history[i].append(0)
        # Normal action selection for step
        if hasattr(agent, "select_action"):
            try:
                action = agent.select_action(agent_obs)
            except Exception:
                action = (
                    np.random.randint(env.action_space)
                    if hasattr(env, "action_space")
                    else 0
                )
        elif hasattr(agent, "act"):
            try:
                action = agent.act(agent_obs)
            except Exception:
                action = (
                    np.random.randint(env.action_space)
                    if hasattr(env, "action_space")
                    else 0
                )
        else:
            action = (
                np.random.randint(env.action_space)
                if hasattr(env, "action_space")
                else 0
            )
        actions.append(action)
    step_result = env.step(actions)
    if isinstance(step_result, tuple) and len(step_result) == 4:
        obs, rewards, done, info = step_result
    else:
        break
    steps += 1
plot_agent_explanations(
    explanation_history, title="Max Q-values or Rule Activations (First Scenario)"
)

# If the first agent is a NeuroFuzzyAgent and explanation includes rule_activations, plot per-rule activations for the first step
if hasattr(agents[0], "explain_action"):
    try:
        explanation = agents[0].explain_action(per_agent_obs[0])
        if "rule_activations" in explanation:
            plot_rule_activations(
                explanation["rule_activations"],
                agent_idx=0,
                title="NeuroFuzzyAgent Rule Activations (First Step)",
            )
    except Exception:
        pass
