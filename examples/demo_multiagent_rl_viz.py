import numpy as np
import matplotlib.pyplot as plt
from src.core.tabular_q_agent import TabularQLearningAgent
from src.env.environment_factory import EnvironmentFactory

# --- Multi-Agent Gridworld Demo with Visualization ---
grid_env = EnvironmentFactory.create(
    "multiagent_gridworld_v2", grid_size=5, n_agents=2, mode="cooperative"
)
agents = [TabularQLearningAgent(n_states=25 * 25, n_actions=4) for _ in range(2)]
reward_history = [[] for _ in agents]
positions_history = [[] for _ in agents]


def obs_to_state(obs):
    return int(obs[0] + 5 * obs[1] + 25 * obs[2] + 125 * obs[3])


n_episodes = 10
for episode in range(n_episodes):
    obs = grid_env.reset()
    total_rewards = [0 for _ in agents]
    positions = [[] for _ in agents]
    for t in range(30):
        states = [obs_to_state(o) for o in obs]
        actions = [agent.act(s) for agent, s in zip(agents, states)]
        next_obs, rewards, done = grid_env.step(actions)
        next_states = [obs_to_state(o) for o in next_obs]
        for i, agent in enumerate(agents):
            agent.observe(rewards[i], next_states[i], done)
            total_rewards[i] += rewards[i]
            positions[i].append(tuple(grid_env.agent_positions[i]))
        obs = next_obs
        if done:
            break
    # Knowledge sharing after each episode
    agents[0].share_knowledge(agents[1], mode="average")
    agents[1].share_knowledge(agents[0], mode="average")
    for i in range(len(agents)):
        reward_history[i].append(total_rewards[i])
        positions_history[i].append(positions[i])

# Plot reward curves
episodes = np.arange(1, n_episodes + 1)
plt.figure(figsize=(8, 4))
for i, rh in enumerate(reward_history):
    plt.plot(episodes, rh, label=f"Agent {i+1}")
plt.plot(episodes, np.mean(reward_history, axis=0), "k--", label="Mean Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Gridworld: Per-Agent and Mean Rewards")
plt.legend()
plt.tight_layout()
plt.show()

# Plot final episode trajectories
plt.figure(figsize=(5, 5))
for i, traj in enumerate(positions_history):
    x, y = zip(*traj[-1])
    plt.plot(x, y, marker="o", label=f"Agent {i+1}")
plt.scatter(*grid_env.target, c="red", marker="*", s=200, label="Target")
plt.xlim(-0.5, grid_env.grid_size - 0.5)
plt.ylim(-0.5, grid_env.grid_size - 0.5)
plt.grid(True)
plt.title("Gridworld: Agent Trajectories (Final Episode)")
plt.legend()
plt.show()

# --- Multi-Agent Resource Collection Demo with Visualization ---
res_env = EnvironmentFactory.create(
    "multiagent_resource", grid_size=5, n_agents=2, n_resources=3, mode="competitive"
)
agents = [TabularQLearningAgent(n_states=None, n_actions=4) for _ in range(2)]
reward_history = [[] for _ in agents]
positions_history = [[] for _ in agents]


def obs_to_state_res(obs):
    return int(sum([v * (5**i) for i, v in enumerate(obs)]))


for episode in range(n_episodes):
    obs = res_env.reset()
    total_rewards = [0 for _ in agents]
    positions = [[] for _ in agents]
    for t in range(30):
        states = [obs_to_state_res(o) for o in obs]
        actions = [agent.act(s) for agent, s in zip(agents, states)]
        next_obs, rewards, done = res_env.step(actions)
        next_states = [obs_to_state_res(o) for o in next_obs]
        for i, agent in enumerate(agents):
            agent.observe(rewards[i], next_states[i], done)
            total_rewards[i] += rewards[i]
            positions[i].append(tuple(res_env.agent_positions[i]))
        obs = next_obs
        if done:
            break
    agents[0].share_knowledge(agents[1], mode="average")
    agents[1].share_knowledge(agents[0], mode="average")
    for i in range(len(agents)):
        reward_history[i].append(total_rewards[i])
        positions_history[i].append(positions[i])

# Plot reward curves
episodes = np.arange(1, n_episodes + 1)
plt.figure(figsize=(8, 4))
for i, rh in enumerate(reward_history):
    plt.plot(episodes, rh, label=f"Agent {i+1}")
plt.plot(episodes, np.mean(reward_history, axis=0), "k--", label="Mean Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Resource Collection: Per-Agent and Mean Rewards")
plt.legend()
plt.tight_layout()
plt.show()

# Plot final episode trajectories
plt.figure(figsize=(5, 5))
for i, traj in enumerate(positions_history):
    x, y = zip(*traj[-1])
    plt.plot(x, y, marker="o", label=f"Agent {i+1}")
# Plot resources collected (approximate, as final positions)
plt.title("Resource Collection: Agent Trajectories (Final Episode)")
plt.xlim(-0.5, res_env.grid_size - 0.5)
plt.ylim(-0.5, res_env.grid_size - 0.5)
plt.grid(True)
plt.legend()
plt.show()
