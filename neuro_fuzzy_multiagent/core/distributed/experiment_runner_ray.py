"""
Distributed experiment runner using Ray for parallel agent-environment execution.
"""

import ray

from neuro_fuzzy_multiagent.core.plugins.registration_utils import (
    get_registered_plugins,
)


@ray.remote
class ExperimentWorker:
    def __init__(self, agent_cls, env_cls, config):
        self.agent = agent_cls(**config.get("agent", {}))
        self.env = env_cls(**config.get("env", {}))

    def run(self, n_episodes):
        results = []
        for _ in range(n_episodes):
            obs = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                action = self.agent.act(obs)
                obs, reward, done, info = self.env.step(action)
                ep_reward += reward
            results.append(ep_reward)
        return results


def run_distributed_experiments(
    agent_name, env_name, config, num_workers=2, episodes_per_worker=5
):
    agents = get_registered_plugins("agent")
    envs = get_registered_plugins("environment")
    agent_cls = agents[agent_name]
    env_cls = envs[env_name]
    ray.init(ignore_reinit_error=True)
    workers = [
        ExperimentWorker.remote(agent_cls, env_cls, config) for _ in range(num_workers)
    ]
    futures = [w.run.remote(episodes_per_worker) for w in workers]
    results = ray.get(futures)
    ray.shutdown()
    return results
