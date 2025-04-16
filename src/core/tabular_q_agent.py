import numpy as np
from .agent import Agent
from .laws import enforce_laws, LawViolation

class TabularQLearningAgent(Agent):
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
            enforce_laws(action, state, category='action')
        except LawViolation:
            action = 0  # fallback to legal action
        self.last_state = observation
        self.last_action = action
        return action

    def observe(self, reward, next_state, done):
        if self.n_states is None:
            q_vals = self.q_table.get(self.last_state, np.zeros(self.n_actions))
            next_q = self.q_table.get(next_state, np.zeros(self.n_actions))
            best_next = np.max(next_q)
            td_target = reward + self.gamma * best_next * (not done)
            td_error = td_target - q_vals[self.last_action]
            q_vals[self.last_action] += self.alpha * td_error
            self.q_table[self.last_state] = q_vals
        else:
            best_next = np.max(self.q_table[next_state])
            td_target = reward + self.gamma * best_next * (not done)
            td_error = td_target - self.q_table[self.last_state, self.last_action]
            self.q_table[self.last_state, self.last_action] += self.alpha * td_error
        if done:
            self.last_state = None
            self.last_action = None
