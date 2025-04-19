"""
distributed_agent_executor.py

Utility for running agent steps in parallel using Ray (distributed, multi-process, multi-node).
"""
import ray

ray.init(ignore_reinit_error=True, log_to_driver=False)

@ray.remote
class RayAgentWrapper:
    def __init__(self, agent):
        self.agent = agent
    def act(self, obs):
        return self.agent.act(obs)
    def get_knowledge(self):
        # Should return Q-table, NN weights, or fuzzy rules depending on agent
        if hasattr(self.agent, 'share_knowledge'):
            return self.agent.share_knowledge()
        return None
    def set_knowledge(self, knowledge):
        # Should update Q-table, NN weights, or fuzzy rules depending on agent
        if hasattr(self.agent, 'integrate_online_knowledge'):
            self.agent.integrate_online_knowledge(knowledge)

def run_agents_distributed(agents, observations):
    """
    Run agent.act(obs) for each agent/observation pair in parallel using Ray actors.
    Args:
        agents: list of agent objects with an 'act' method
        observations: list of observations (one per agent)
    Returns:
        actions: list of actions (same order as agents)
    """
    ray_agents = [RayAgentWrapper.remote(agent) for agent in agents]
    futures = [ray_agents[i].act.remote(observations[i]) for i in range(len(agents))]
    return ray.get(futures)
