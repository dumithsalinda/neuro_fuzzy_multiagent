import numpy as np

from neuro_fuzzy_multiagent.core.dqn_agent import DQNAgent
from neuro_fuzzy_multiagent.core.tabular_q_agent import TabularQLearningAgent
from neuro_fuzzy_multiagent.env.simple_env import SimpleContinuousEnv, SimpleDiscreteEnv

# --- Tabular Q-Learning Demo ---
print("=== Tabular Q-Learning Agent in Discrete Env ===")
env = SimpleDiscreteEnv(n_states=5, n_actions=2)
agent = TabularQLearningAgent(n_states=5, n_actions=2)

for episode in range(10):
    state = env.reset()
    total_reward = 0
    for t in range(20):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.observe(reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    print(f"Episode {episode+1}: Total Reward = {total_reward}")

# --- DQN Agent Demo ---
print("\n=== DQN Agent in Continuous Env ===")
env = SimpleContinuousEnv()
agent = DQNAgent(state_dim=2, action_dim=4)

for episode in range(5):
    state = env.reset()
    total_reward = 0
    for t in range(30):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.observe(reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}")
