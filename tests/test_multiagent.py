"""
test_multiagent.py

Tests for multiagent simulation, agent-environment interface, and reward aggregation.
"""

import numpy as np
from core.agent import Agent
from environment.abstraction import SimpleEnvironment
from environment.transfer_learning import FeatureExtractor

def test_multiagent_simulation_runs():
    np.random.seed(2024)
    num_agents = 2
    episodes = 3
    steps_per_episode = 5
    agents = []
    envs = []
    feat_extractors = []
    rewards = np.zeros((num_agents, episodes, steps_per_episode))
    for _ in range(num_agents):
        env = SimpleEnvironment(dim=3)
        model = lambda x: x  # Identity model
        feat_extractor = FeatureExtractor(input_dim=3, output_dim=3)
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
                reward = -np.linalg.norm(action)
                agent.observe(reward)
                rewards[i, ep, step] = reward
    # Check reward array shape
    assert rewards.shape == (num_agents, episodes, steps_per_episode)
    # Check that at least one reward is nonzero for each agent
    assert np.any(rewards != 0, axis=(1,2)).all()

def test_agent_reset_and_reuse():
    env = SimpleEnvironment(dim=2)
    model = lambda x: x
    feat_extractor = FeatureExtractor(input_dim=2, output_dim=2)
    agent = Agent(model)
    state = env.reset()
    features = feat_extractor.extract(state)
    action = agent.act(features)
    env.step(action)
    agent.observe(-1.0)
    agent.reset()
    assert agent.total_reward == 0
    assert agent.last_action is None
    assert agent.last_observation is None
