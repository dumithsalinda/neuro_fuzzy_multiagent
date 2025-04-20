"""
Distributed Curriculum Batch Experiment Runner using Ray
Each curriculum stage yields harder scenarios for distributed benchmarking.
"""
import ray
import numpy as np
from src.core.agent_factory import create_agent_from_config
from src.core.experiment_manager import ExperimentManager
from src.core.curriculum_scenario_generator import CurriculumScenarioGenerator
from src.env.environment_factory import EnvironmentFactory

ENV_MAP = {
    "Gridworld": "multiagent_gridworld",
    "IoTSensorFusionEnv": "iot_sensor_fusion"
}
AGENT_CFG_MAP = {
    "DQNAgent": "examples/agent_config_dqn.yaml",
    "NeuroFuzzyAgent": "examples/agent_config_example.yaml"
}

@ray.remote
def run_scenario(scenario):
    np.random.seed(scenario["seed"])
    env_key = ENV_MAP[scenario["env"]]
    env = EnvironmentFactory.create(env_key, n_agents=scenario["agent_count"])
    agent_cfg_path = AGENT_CFG_MAP[scenario["agent_type"]]
    agents = [create_agent_from_config(
        __import__('yaml').safe_load(open(agent_cfg_path))) for _ in range(scenario["agent_count"])]
    obs = env.reset()
    total_rewards = [0 for _ in range(scenario["agent_count"])]
    done = False
    steps = 0
    while not done and steps < 50:
        per_agent_obs = [obs for _ in range(scenario["agent_count"])]
        actions = []
        for i, agent in enumerate(agents):
            agent_obs = per_agent_obs[i]
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
    base_grid = {
        "agent_type": ["DQNAgent", "NeuroFuzzyAgent"],
        "env": ["Gridworld", "IoTSensorFusionEnv"],
        "seed": [42, 123],
    }
    curriculum_steps = [
        {"agent_count": 2, "n_obstacles": 1},
        {"agent_count": 3, "n_obstacles": 2},
        {"agent_count": 4, "n_obstacles": 3},
    ]
    generator = CurriculumScenarioGenerator(base_grid, curriculum_steps)
    mgr = ExperimentManager(log_dir="experiments")
    for stage, scenario in enumerate(generator.curriculum()):
        print("[Curriculum Stage {}] Scenario: {}".format(stage, scenario))
        future = run_scenario.remote(scenario)
        result = ray.get(future)
        mgr.log_results(mgr.start_run(result["scenario"]), {"mean_reward": result["mean_reward"], "std_reward": result["std_reward"], "agent_count": result["scenario"]["agent_count"]})
        print("[Distributed Curriculum] {} | mean_reward={:.2f}".format(result['scenario'], result['mean_reward']))
    print("Curriculum run complete.")
