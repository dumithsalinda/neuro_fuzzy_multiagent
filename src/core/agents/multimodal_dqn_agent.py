import torch
import torch.nn as nn
from .agent import Agent
from .fusion import FusionNetwork


class MultiModalDQNAgent(Agent):
    def explain_action(self, features):
        import torch

        features = [torch.FloatTensor(f).unsqueeze(0).to(self.device) for f in features]
        with torch.no_grad():
            q_values = self.fusion_net(features).cpu().numpy().flatten()
        action = int(q_values.argmax())
        return {
            "q_values": q_values.tolist(),
            "chosen_action": action,
            "epsilon": self.epsilon,
        }

    """
    DQN agent that takes multi-modal features (e.g., text + image) as input.
    Features should be provided as a list of tensors (one per modality).
    """

    def __init__(self, input_dims, action_dim, alpha=1e-3, gamma=0.99, epsilon=0.1):
        super().__init__(model=None)
        self.input_dims = input_dims  # list of input dims per modality
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = torch.device("cpu")
        self.fusion_net = FusionNetwork(
            input_dims, hidden_dim=128, output_dim=action_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.fusion_net.parameters(), lr=alpha)
        self.memory = []  # (features, action, reward, next_features, done)
        self.batch_size = 32
        self.last_features = None
        self.last_action = None

    def act(self, features, state=None):
        # features: list of numpy arrays or tensors
        features = [torch.FloatTensor(f).unsqueeze(0).to(self.device) for f in features]
        if torch.rand(1).item() < self.epsilon:
            action = torch.randint(self.action_dim, (1,)).item()
        else:
            with torch.no_grad():
                q_values = self.fusion_net(features)
                action = q_values.argmax().item()
        self.last_features = [f.detach().cpu().numpy() for f in features]
        self.last_action = action
        return action

    def observe(self, reward, next_features, done):
        self.memory.append(
            (self.last_features, self.last_action, reward, next_features, done)
        )
        if len(self.memory) >= self.batch_size:
            self.train_step()
        if done:
            self.last_features = None

    def train_step(self):
        # Simple DQN-style batch update (stub, not prioritized)
        import random

        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        features, actions, rewards, next_features, dones = zip(*batch)
        features = [
            torch.FloatTensor([f[i] for f in features]).to(self.device)
            for i in range(len(self.input_dims))
        ]
        next_features = [
            torch.FloatTensor([nf[i] for nf in next_features]).to(self.device)
            for i in range(len(self.input_dims))
        ]
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        q_values = self.fusion_net(features).gather(1, actions)
        with torch.no_grad():
            next_q = self.fusion_net(next_features).max(1, keepdim=True)[0]
            target = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
