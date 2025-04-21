import ray
import time
import importlib

class DistributedExperimentOrchestrator:
    """
    Orchestrates distributed experiments using Ray.
    Launches multiple agents/environments as Ray actors, collects results, and supports dynamic scaling.
    """
    def __init__(self, agent_class, env_class, agent_kwargs=None, env_kwargs=None, num_agents=4):
        self.agent_class = agent_class
        self.env_class = env_class
        self.agent_kwargs = agent_kwargs or {}
        self.env_kwargs = env_kwargs or {}
        self.num_agents = num_agents
        self.agents = []
        self.envs = []

    def launch(self):
        ray.init(ignore_reinit_error=True)
        RemoteAgent = ray.remote(self.agent_class)
        RemoteEnv = ray.remote(self.env_class)
        self.envs = [RemoteEnv.remote(**self.env_kwargs) for _ in range(self.num_agents)]
        self.agents = [RemoteAgent.remote(**self.agent_kwargs) for _ in range(self.num_agents)]

    def run_episode(self, steps=100):
        results = []
        for i in range(self.num_agents):
            agent = self.agents[i]
            env = self.envs[i]
            obs = ray.get(env.reset.remote())
            total_reward = 0
            for _ in range(steps):
                action = ray.get(agent.act.remote(obs))
                obs, reward, done, info = ray.get(env.step.remote(action))
                ray.get(agent.observe.remote(reward, obs, done))
                total_reward += reward
                if done:
                    break
            results.append(total_reward)
        return results

    def shutdown(self):
        ray.shutdown()
