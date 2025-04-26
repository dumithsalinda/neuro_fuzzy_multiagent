"""
test_benchmark_multiagent_with_neurofuzzy.py

Test multiagent benchmarking with NeuroFuzzyAgent and baseline agents.
Ensures simulation runs, rewards are collected, and shapes are correct.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from src.core.agents.agent import Agent, NeuroFuzzyAgent
from src.environment.abstraction import SimpleEnvironment
from src.environment.transfer_learning import FeatureExtractor


def test_benchmark_with_neurofuzzy_agent():
    np.random.seed(2025)
    num_agents = 2
    episodes = 2
    steps_per_episode = 3
    agents = []
    envs = []
    feat_extractors = []
    # Agent 0: baseline random, Agent 1: NeuroFuzzyAgent
    agents.append(Agent(model=None))
    nn_config = {"input_dim": 1, "output_dim": 1, "hidden_dim": 4}
    agents.append(NeuroFuzzyAgent(nn_config, None))
    for _ in range(num_agents):
        env = SimpleEnvironment(dim=1)
        feat_extractor = FeatureExtractor(input_dim=1, output_dim=1)
        envs.append(env)
        feat_extractors.append(feat_extractor)
    rewards = np.zeros((num_agents, episodes, steps_per_episode))
    for ep in range(episodes):
        for agent in agents:
            agent.reset()
        for step in range(steps_per_episode):
            for i, (agent, env, extractor) in enumerate(
                zip(agents, envs, feat_extractors)
            ):
                state = env.perceive()
                features = extractor.extract(state)
                # For baseline agent, use random action; for NeuroFuzzyAgent, use act()
                if isinstance(agent, NeuroFuzzyAgent):
                    action = agent.act(features)
                else:
                    action = np.random.randn(*features.shape)
                env.step(action)
                reward = -np.linalg.norm(action)
                agent.observe(reward)
                rewards[i, ep, step] = reward
    assert rewards.shape == (num_agents, episodes, steps_per_episode)
    assert np.any(rewards != 0)
