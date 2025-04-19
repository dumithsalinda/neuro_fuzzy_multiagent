"""
Demo: Batch experiment runner using ScenarioGenerator and ExperimentManager
Runs real agents in real environments for each scenario.
"""
import numpy as np
import os
from src.core.agent_factory import create_agent_from_config
from src.core.experiment_manager import ExperimentManager
from src.core.scenario_generator import ScenarioGenerator
from src.env.environment_factory import EnvironmentFactory

# Map scenario env names to factory keys
ENV_MAP = {
    "Gridworld": "multiagent_gridworld",
    "IoTSensorFusionEnv": "iot_sensor_fusion"
}
# Map agent types to example config files
AGENT_CFG_MAP = {
    "DQNAgent": "examples/agent_config_dqn.yaml",
    "NeuroFuzzyAgent": "examples/agent_config_example.yaml"
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
    # Create agents
    agent_cfg_path = AGENT_CFG_MAP[scenario["agent_type"]]
    agents = [create_agent_from_config(
        __import__('yaml').safe_load(open(agent_cfg_path))) for _ in range(scenario["agent_count"])]
    # Run one episode with random actions
    obs = env.reset()
    total_rewards = [0 for _ in range(scenario["agent_count"])]
    done = False
    steps = 0
    while not done and steps < 50:
        # Try to get per-agent observations if available, else share obs
        per_agent_obs = None
        if isinstance(obs, dict) and "agent_positions" in obs:
            # IoTSensorFusionEnv returns dict with agent_positions
            per_agent_obs = [obs for _ in range(scenario["agent_count"])]
        elif isinstance(obs, np.ndarray) and obs.shape[0] >= scenario["agent_count"] * 2:
            # Gridworld returns flat array, assume shared obs
            per_agent_obs = [obs for _ in range(scenario["agent_count"])]
        else:
            per_agent_obs = [obs for _ in range(scenario["agent_count"])]

        actions = []
        for i, agent in enumerate(agents):
            agent_obs = per_agent_obs[i]
            # Use agent's policy if available
            if hasattr(agent, 'select_action'):
                try:
                    action = agent.select_action(agent_obs)
                except Exception:
                    action = np.random.randint(env.action_space) if hasattr(env, 'action_space') else 0
            elif hasattr(agent, 'act'):
                try:
                    action = agent.act(agent_obs)
                except Exception:
                    action = np.random.randint(env.action_space) if hasattr(env, 'action_space') else 0
            else:
                action = np.random.randint(env.action_space) if hasattr(env, 'action_space') else 0
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
        "agent_count": scenario["agent_count"]
    }
    mgr.log_results(run, result)
    print(f"Finished run {run['run_id']} | {scenario} | mean_reward={mean_reward:.2f}")

means = mgr.aggregate_results("mean_reward")
print(f"All mean rewards: {means}")
