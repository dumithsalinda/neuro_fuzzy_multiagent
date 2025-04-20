"""
multiagent_simulation.py

Demonstrates a simple multiagent scenario with multiple agents interacting with their own environments.
Agents use random or greedy policies and accumulate rewards.
"""

import numpy as np
from src.core.agents.agent import Agent
from src.environment.abstraction import SimpleEnvironment
from src.environment.transfer_learning import FeatureExtractor

class DummyModel:
    def forward(self, x):
        return x * 2

def run_multiagent_simulation(num_agents=3, steps=10):
    np.random.seed(42)
    agents = []
    envs = []
    feat_extractors = []
    for _ in range(num_agents):
        env = SimpleEnvironment(dim=4)
        model = DummyModel()
        feat_extractor = FeatureExtractor(input_dim=4, output_dim=2)
        agent = Agent(model)
        agents.append(agent)
        envs.append(env)
        feat_extractors.append(feat_extractor)
    for step in range(steps):
        for i, (agent, env, extractor) in enumerate(zip(agents, envs, feat_extractors)):
            state = env.perceive()
            features = extractor.extract(state)
            action = agent.act(features)
            next_state, reward = env.step(action)
            agent.observe(reward)
    for i, agent in enumerate(agents):
        print(f"Agent {i} total reward: {agent.total_reward}")

if __name__ == "__main__":
    run_multiagent_simulation()
