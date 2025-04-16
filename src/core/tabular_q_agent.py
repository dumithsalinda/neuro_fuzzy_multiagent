import numpy as np
from .agent import Agent
from .laws import enforce_laws, LawViolation

class TabularQLearningAgent(Agent):
    """
    Tabular Q-Learning Agent for discrete state and action spaces.
    """
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(model=None)
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))
        self.last_state = None
        self.last_action = None

    def act(self, observation, state=None):
        # Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.q_table[observation])
        try:
            enforce_laws(action, state, category='action')
        except LawViolation:
            action = 0  # fallback to legal action
        self.last_state = observation
        self.last_action = action
        return action

    def observe(self, reward, next_state, done):
        # Q-learning update
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next * (not done)
        td_error = td_target - self.q_table[self.last_state, self.last_action]
        self.q_table[self.last_state, self.last_action] += self.alpha * td_error
        if done:
            self.last_state = None
            self.last_action = None
