import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .agent import Agent
from .laws import enforce_laws, LawViolation

class QNetwork(nn.Module):
    """
    Q-Network for DQNAgent. state_dim should match the feature vector dimension for the agent's input type (e.g., 768 for BERT, 512 for ResNet18).
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent(Agent):
    def explain_action(self, observation):
        import torch
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor).cpu().numpy().flatten()
        action = int(q_values.argmax())
        return {
            "q_values": q_values.tolist(),
            "chosen_action": action,
            "epsilon": self.epsilon
        }

    """
    Deep Q-Learning Agent for continuous or large state spaces.
    """
    def __init__(self, state_dim, action_dim, alpha=1e-3, gamma=0.99, epsilon=0.1):
        super().__init__(model=None)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = torch.device('cpu')
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=alpha)
        self.memory = []  # (state, action, reward, next_state, done)
        self.batch_size = 32
        self.last_state = None
        self.last_action = None

    def act(self, observation, state=None):
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.q_net(obs_tensor)
                action = q_values.argmax().item()
        try:
            enforce_laws(action, state, category='action')
        except LawViolation:
            action = 0
        self.last_state = observation
        self.last_action = action
        return action

    def observe(self, reward, next_state, done):
        # Store experience
        self.memory.append((self.last_state, self.last_action, reward, next_state, done))
        if len(self.memory) >= self.batch_size:
            self.train_step()
        if done:
            self.last_state = None
            self.last_action = None

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)
        # max_a' Q(s',a')
        with torch.no_grad():
            next_q = self.q_net(next_states).max(1, keepdim=True)[0]
            target = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
