import numpy as np

from neuro_fuzzy_multiagent.core.agents.agent import Agent
from neuro_fuzzy_multiagent.core.plugins.registration_utils import register_plugin


@register_plugin("agent")
class MetaAgent(Agent):
    """
    MetaAgent adapts its own learning algorithm or architecture based on performance metrics.
    It can switch between different agent models (DQN, NeuroFuzzy, etc.) or tune hyperparameters on the fly.
    """

    def __init__(
        self, candidate_agents, selection_strategy=None, perf_metric="reward", window=10
    ):
        """
        candidate_agents: list of (agent_class, config_dict)
        selection_strategy: function(perf_history) -> index of agent to use
        perf_metric: which metric to optimize (default: reward)
        window: sliding window size for performance evaluation
        """
        # Start with the first candidate agent
        super().__init__(model=None)
        self.candidate_agents = [cls(**cfg) for cls, cfg in candidate_agents]
        self.active_idx = 0
        self.active_agent = self.candidate_agents[self.active_idx]
        self.perf_history = [[] for _ in self.candidate_agents]
        self.perf_metric = perf_metric
        self.window = window
        self.selection_strategy = selection_strategy or self.default_selection_strategy
        self._explore_phase = True
        self._explore_agent_idx = 0
        self._explore_steps = 0

    def act(self, observation, state=None):
        return self.active_agent.act(observation, state)

    def observe(self, reward, next_state, done):
        self.active_agent.observe(reward, next_state, done)
        self.perf_history[self.active_idx].append(reward)
        # Only keep window size
        if len(self.perf_history[self.active_idx]) > self.window:
            self.perf_history[self.active_idx] = self.perf_history[self.active_idx][
                -self.window :
            ]
        if self._explore_phase:
            self._explore_steps += 1
            if self._explore_steps >= self.window:
                self._explore_agent_idx += 1
                self._explore_steps = 0
                if self._explore_agent_idx < len(self.candidate_agents):
                    self.active_idx = self._explore_agent_idx
                    self.active_agent = self.candidate_agents[self.active_idx]
                else:
                    # End exploration, pick best
                    avg_perf = [
                        np.mean(h[-self.window :]) if h else -np.inf
                        for h in self.perf_history
                    ]
                    self.active_idx = self.selection_strategy(avg_perf)
                    self.active_agent = self.candidate_agents[self.active_idx]
                    self._explore_phase = False
        else:
            self.maybe_switch_agent()

    def maybe_switch_agent(self):
        # Evaluate all agents' recent performance
        avg_perf = [
            np.mean(h[-self.window :]) if h else -np.inf for h in self.perf_history
        ]
        new_idx = self.selection_strategy(avg_perf)
        if new_idx != self.active_idx:
            self.active_idx = new_idx
            self.active_agent = self.candidate_agents[self.active_idx]

    @staticmethod
    def default_selection_strategy(avg_perf):
        # Choose agent with highest average performance
        return int(np.argmax(avg_perf))

    def reset(self):
        for agent in self.candidate_agents:
            agent.reset()
        self.active_idx = 0
        self.active_agent = self.candidate_agents[0]
        self.perf_history = [[] for _ in self.candidate_agents]

    def __getattr__(self, name):
        # Proxy attribute access to the active agent
        return getattr(self.active_agent, name)
