"""
benchmark_multiagent.py

Benchmarking and visualization for multiagent scenarios.
Runs multiple agents for several episodes, collects rewards, and plots learning curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.core.agents.agent import Agent, NeuroFuzzyAgent
from src.environment.abstraction import SimpleEnvironment
from src.environment.transfer_learning import FeatureExtractor

def run_benchmark(agent_types, episodes=20, steps_per_episode=30):
    np.random.seed(123)
    agents = []
    envs = []
    feat_extractors = []
    num_agents = len(agent_types)
    rewards = np.zeros((num_agents, episodes, steps_per_episode))
    for agent_type in agent_types:
        env = SimpleEnvironment(dim=1)
        feat_extractor = FeatureExtractor(input_dim=1, output_dim=1)
        if agent_type == 'neurofuzzy':
            nn_config = {"input_dim": 1, "output_dim": 1, "hidden_dim": 4}
            agent = NeuroFuzzyAgent(nn_config, None)
        else:
            agent = Agent(model=None)
        agents.append(agent)
        envs.append(env)
        feat_extractors.append(feat_extractor)
    for ep in range(episodes):
        for agent in agents:
            agent.reset()
        for step in range(steps_per_episode):
            for i, (agent, env, extractor) in enumerate(zip(agents, envs, feat_extractors)):
                state = env.perceive()
                features = extractor.extract(state)
                if isinstance(agent, NeuroFuzzyAgent):
                    action = agent.act(features)
                else:
                    action = np.random.randn(*features.shape)
                env.step(action)
                reward = -np.linalg.norm(action)
                agent.observe(reward)
                rewards[i, ep, step] = reward
    return rewards

def plot_learning_curves(rewards, agent_types):
    num_agents, episodes, steps = rewards.shape
    avg_rewards = rewards.mean(axis=1)  # (num_agents, steps)
    plt.figure(figsize=(8, 5))
    for i in range(num_agents):
        label = f"Agent {i} ({agent_types[i]})"
        plt.plot(avg_rewards[i], label=label)
    plt.xlabel("Step")
    plt.ylabel("Average Reward per Episode")
    plt.title("Multiagent Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    agent_types = ['random', 'neurofuzzy']
    rewards = run_benchmark(agent_types, episodes=20, steps_per_episode=30)
    plot_learning_curves(rewards, agent_types)

