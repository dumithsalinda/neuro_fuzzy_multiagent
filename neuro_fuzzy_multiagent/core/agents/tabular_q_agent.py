import numpy as np

from neuro_fuzzy_multiagent.core.agents.agent import Agent
from neuro_fuzzy_multiagent.core.agents.laws import LawViolation, enforce_laws
from neuro_fuzzy_multiagent.core.plugins.registration_utils import register_plugin


@register_plugin("agent")
class TabularQLearningAgent(Agent):
    def explain_action(self, observation):
        import numpy as np

        if self.n_states is None:
            q_vals = self.q_table.get(observation, np.zeros(self.n_actions))
        else:
            q_vals = self.q_table[observation]
        action = int(np.argmax(q_vals))
        return {
            "q_values": q_vals.tolist(),
            "chosen_action": action,
            "epsilon": self.epsilon,
        }

    """
    Tabular Q-Learning Agent for discrete state and action spaces.
    If n_states is None, uses a dict-based Q-table for large/sparse spaces.
    """

    def __init__(self, n_states=None, n_actions=2, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(model=None)
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        if n_states is None:
            self.q_table = {}  # dict-based for large state spaces
        else:
            self.q_table = np.zeros((n_states, n_actions))
        self.last_state = None
        self.last_action = None

    def act(self, observation, state=None):
        # Epsilon-greedy policy
        if self.n_states is None:
            q_vals = self.q_table.get(observation, np.zeros(self.n_actions))
        else:
            q_vals = self.q_table[observation]
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(q_vals)
        try:
            enforce_laws(action, state, category="action")
        except LawViolation:
            action = 0  # fallback to legal action
        self.last_state = observation
        self.last_action = action
        return action

    def share_knowledge(self, other_agent, mode="average"):
        """
        Share Q-table knowledge with another agent.
        mode: 'copy' (overwrite self), 'average' (elementwise mean)
        """
        if self.n_states is None:
            # Dict-based Q-table
            all_keys = set(self.q_table.keys()).union(other_agent.q_table.keys())
            for k in all_keys:
                q1 = self.q_table.get(k, np.zeros(self.n_actions))
                q2 = other_agent.q_table.get(k, np.zeros(self.n_actions))
                if mode == "copy":
                    self.q_table[k] = q2.copy()
                elif mode == "average":
                    self.q_table[k] = (q1 + q2) / 2.0
        else:
            if mode == "copy":
                self.q_table = other_agent.q_table.copy()
            elif mode == "average":
                self.q_table = (self.q_table + other_agent.q_table) / 2.0
