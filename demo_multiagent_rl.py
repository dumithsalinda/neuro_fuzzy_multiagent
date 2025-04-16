import numpy as np
from src.core.tabular_q_agent import TabularQLearningAgent
from src.env.multiagent_gridworld import MultiAgentGridworldEnv
from src.env.multiagent_resource import MultiAgentResourceEnv

# --- Multi-Agent Gridworld Demo ---
print("=== Multi-Agent Gridworld (Cooperative) ===")
grid_env = MultiAgentGridworldEnv(grid_size=5, n_agents=2, mode="cooperative")
agents = [TabularQLearningAgent(n_states=25*25, n_actions=4) for _ in range(2)]

def obs_to_state(obs):
    # flatten (x, y, tx, ty) into a discrete state
    return int(obs[0] + 5*obs[1] + 25*obs[2] + 125*obs[3])

for episode in range(5):
    obs = grid_env.reset()
    total_rewards = [0 for _ in agents]
    for t in range(30):
        states = [obs_to_state(o) for o in obs]
        actions = [agent.act(s) for agent, s in zip(agents, states)]
        next_obs, rewards, done = grid_env.step(actions)
        next_states = [obs_to_state(o) for o in next_obs]
        for i, agent in enumerate(agents):
            agent.observe(rewards[i], next_states[i], done)
            total_rewards[i] += rewards[i]
        obs = next_obs
        if done:
            break
    print(f"Episode {episode+1}: Rewards = {total_rewards}")

# --- Multi-Agent Resource Collection Demo ---
print("\n=== Multi-Agent Resource Collection (Competitive) ===")
res_env = MultiAgentResourceEnv(grid_size=5, n_agents=2, n_resources=3, mode="competitive")
agents = [TabularQLearningAgent(n_states=None, n_actions=4) for _ in range(2)]

def obs_to_state_res(obs):
    # flatten (x, y, rx1, ry1, rx2, ry2, ...) into a discrete state
    return int(sum([v * (5 ** i) for i, v in enumerate(obs)]))

for episode in range(5):
    obs = res_env.reset()
    total_rewards = [0 for _ in agents]
    for t in range(30):
        states = [obs_to_state_res(o) for o in obs]
        actions = [agent.act(s) for agent, s in zip(agents, states)]
        next_obs, rewards, done = res_env.step(actions)
        next_states = [obs_to_state_res(o) for o in next_obs]
        for i, agent in enumerate(agents):
            agent.observe(rewards[i], next_states[i], done)
            total_rewards[i] += rewards[i]
        obs = next_obs
        if done:
            break
    print(f"Episode {episode+1}: Rewards = {total_rewards}")
