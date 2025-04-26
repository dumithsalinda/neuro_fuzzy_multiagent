"""
Distributed Multi-Agent Environment Runner using Ray
Each agent runs as a Ray actor; environment coordinates actions and observations.
"""

import numpy as np
import ray

from neuro_fuzzy_multiagent.core.management.agent_factory import (
    create_agent_from_config,
)
from neuro_fuzzy_multiagent.env.environment_factory import EnvironmentFactory

# Map agent types to example config files
AGENT_CFG_MAP = {
    "DQNAgent": "examples/agent_config_dqn.yaml",
    "NeuroFuzzyAgent": "examples/agent_config_example.yaml",
}


@ray.remote
class RemoteAgent:
    def __init__(self, agent_type):
        agent_cfg_path = AGENT_CFG_MAP[agent_type]
        self.agent = create_agent_from_config(
            __import__("yaml").safe_load(open(agent_cfg_path))
        )
        self.inbox = []

    def compute_action(self, obs):
        if hasattr(self.agent, "select_action"):
            try:
                return self.agent.select_action(obs)
            except Exception:
                return 0
        elif hasattr(self.agent, "act"):
            try:
                return self.agent.act(obs)
            except Exception:
                return 0
        else:
            return 0

    def send_message(self, msg):
        self.inbox.append(msg)

    def receive_messages(self):
        msgs = self.inbox[:]
        self.inbox.clear()
        return msgs


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    env = EnvironmentFactory.create("multiagent_gridworld", n_agents=4)
    agents = [RemoteAgent.remote("DQNAgent") for _ in range(4)]
    obs = env.reset()
    done = False
    steps = 0
    total_rewards = [0 for _ in range(4)]
    while not done and steps < 50:
        per_agent_obs = [obs for _ in range(4)]
        actions = ray.get(
            [
                agent.compute_action.remote(per_agent_obs[i])
                for i, agent in enumerate(agents)
            ]
        )
        # Broadcast actions to all agents
        for i, agent in enumerate(agents):
            for j, action in enumerate(actions):
                if i != j:
                    agent.send_message.remote({"from": j, "action": action})
        # Each agent receives its messages
        messages = ray.get([agent.receive_messages.remote() for agent in agents])
        # (Optional) Print agent messages for demonstration
        for idx, msgs in enumerate(messages):
            print(f"Agent {idx} received messages: {msgs}")
        step_result = env.step(actions)
        if isinstance(step_result, tuple) and len(step_result) == 4:
            obs, rewards, done, info = step_result
            if isinstance(rewards, list):
                for i, r in enumerate(rewards):
                    total_rewards[i] += r
            else:
                for i in range(4):
                    total_rewards[i] += rewards
        else:
            break
        steps += 1
    print(f"Distributed agent run complete. Mean reward: {np.mean(total_rewards):.2f}")
