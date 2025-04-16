"""
benchmark_multiagent.py

Benchmarking and visualization for multiagent scenarios.
Runs multiple agents for several episodes, collects rewards, and plots learning curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.core.agent import Agent
from src.environment.abstraction import SimpleEnvironment
from src.environment.transfer_learning import FeatureExtractor

class DummyModel:
    def forward(self, x):
        return x * 2

def run_benchmark(num_agents=3, episodes=20, steps_per_episode=30):
    np.random.seed(123)
    agents = []
    envs = []
    feat_extractors = []
    rewards = np.zeros((num_agents, episodes, steps_per_episode))
    for _ in range(num_agents):
        env = SimpleEnvironment(dim=4)
        model = DummyModel()
        feat_extractor = FeatureExtractor(input_dim=4, output_dim=4)  # output_dim matches env dim
        agent = Agent(model)
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
                action = agent.act(features)
                next_state = env.step(action)
                # Define reward as negative L2 norm of action (proxy for minimal effort)
                reward = -np.linalg.norm(action)
                agent.observe(reward)
                rewards[i, ep, step] = reward
    return rewards

def plot_learning_curves(rewards):
    num_agents, episodes, steps = rewards.shape
    avg_rewards = rewards.mean(axis=1)  # (num_agents, steps)
    plt.figure(figsize=(8, 5))
    for i in range(num_agents):
        plt.plot(avg_rewards[i], label=f"Agent {i}")
    plt.xlabel("Step")
    plt.ylabel("Average Reward per Episode")
    plt.title("Multiagent Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rewards = run_benchmark(num_agents=3, episodes=20, steps_per_episode=30)
    plot_learning_curves(rewards)
