"""
Distributed Batch Experiment Runner using Ray
Runs each experiment scenario in parallel as a Ray task.
"""

import numpy as np
import ray

from neuro_fuzzy_multiagent.core.experiment_manager import ExperimentManager
from neuro_fuzzy_multiagent.core.management.agent_factory import (
    create_agent_from_config,
)
from neuro_fuzzy_multiagent.core.scenario_generator import ScenarioGenerator
from neuro_fuzzy_multiagent.env.environment_factory import EnvironmentFactory

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


@ray.remote
def run_scenario(scenario):
    np.random.seed(scenario["seed"])
    env_key = ENV_MAP[scenario["env"]]
    env = EnvironmentFactory.create(env_key, n_agents=scenario["agent_count"])
    agent_cfg_path = AGENT_CFG_MAP[scenario["agent_type"]]
    agents = [
        create_agent_from_config(__import__("yaml").safe_load(open(agent_cfg_path)))
        for _ in range(scenario["agent_count"])
    ]
    obs = env.reset()
    total_rewards = [0 for _ in range(scenario["agent_count"])]
    done = False
    steps = 0
    while not done and steps < 50:
        per_agent_obs = [obs for _ in range(scenario["agent_count"])]
        actions = []
        for i, agent in enumerate(agents):
            agent_obs = per_agent_obs[i]
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
    return {"scenario": scenario, "mean_reward": mean_reward, "std_reward": std_reward}


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    param_grid = {
        "agent_type": ["DQNAgent", "NeuroFuzzyAgent"],
        "agent_count": [2, 4],
        "env": ["Gridworld", "IoTSensorFusionEnv"],
        "seed": [42, 123],
    }
    scenarios = ScenarioGenerator(param_grid).grid_search()
    mgr = ExperimentManager(log_dir="experiments")
    futures = [run_scenario.remote(scenario) for scenario in scenarios]
    results = ray.get(futures)
    for result in results:
        mgr.log_results(
            mgr.start_run(result["scenario"]),
            {
                "mean_reward": result["mean_reward"],
                "std_reward": result["std_reward"],
                "agent_count": result["scenario"]["agent_count"],
            },
        )
        print(
            f"[Distributed] {result['scenario']} | mean_reward={result['mean_reward']:.2f}"
        )
    means = mgr.aggregate_results("mean_reward")
    print(f"All distributed mean rewards: {means}")
