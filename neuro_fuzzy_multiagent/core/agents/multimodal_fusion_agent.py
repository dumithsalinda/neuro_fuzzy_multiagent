import torch
import torch.nn as nn

from neuro_fuzzy_multiagent.core.neural_networks.fusion import FusionNetwork
from neuro_fuzzy_multiagent.core.plugins.registration_utils import register_plugin


@register_plugin("agent")
class MultiModalFusionAgent:
    """
    Agent that accepts multiple modalities as input and uses a fusion network for policy/Q-value computation.
    Easily extensible: add new fusion strategies in FusionNetwork and update here.
    TODO: Add support for training (currently eval-only), more modalities, and advanced fusion methods (e.g., attention, gating).
    """

    def __init__(
        self,
        input_dims,
        hidden_dim,
        output_dim,
        fusion_type="concat",
        lr=1e-3,
        gamma=0.99,
        **kwargs
    ):
        """
        Args:
            input_dims (list[int]): List of input dimensions for each modality (e.g., [img_dim, txt_dim]).
            hidden_dim (int): Hidden layer size for fusion network.
            output_dim (int): Number of possible actions.
            fusion_type (str): Fusion strategy ('concat', 'attention', 'gating').
            lr (float): Learning rate for optimizer.
            gamma (float): Discount factor for Q-learning.
        """
        self.model = FusionNetwork(
            input_dims, hidden_dim, output_dim, fusion_type=fusion_type
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.train_mode = False
        self.model.eval()
        self.extra_args = kwargs

    def set_train_mode(self, is_train: bool):
        self.train_mode = is_train
        if is_train:
            self.model.train()
        else:
            self.model.eval()

    def observe(self, obs_list, action, reward, next_obs_list, done):
        """
        Online Q-learning update for a single step.
        Args:
            obs_list: list of modality features for current state
            action: int, action taken
            reward: float
            next_obs_list: list of modality features for next state
            done: bool
        """
        if not hasattr(self, "loss_history"):
            from collections import deque

            self.loss_history = deque(maxlen=200)
        self.set_train_mode(True)
        obs = [
            torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
            for x in obs_list
        ]
        next_obs = [
            torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
            for x in next_obs_list
        ]
        qvals = self.model(obs)  # shape: [1, n_actions]
        qval = qvals[0, action]
        with torch.no_grad():
            qvals_next = self.model(next_obs)
            max_q_next = torch.max(qvals_next, dim=-1)[0].item()
            target = reward + (0 if done else self.gamma * max_q_next)
            target = torch.tensor(target, dtype=torch.float32, device=self.device)
        loss = self.loss_fn(qval, target)
        self.loss_history.append(float(loss.item()))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.set_train_mode(False)

    def act(self, obs_list):
        """
        Select action given a list of modality features.
        Args:
            obs_list (list[np.ndarray or torch.Tensor]): Features for each modality.
        Returns:
            int: Selected action (argmax Q-value)
        """
        features = [
            torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
            for x in obs_list
        ]
        with torch.no_grad():
            qvals = self.model(features)
        action = torch.argmax(qvals, dim=-1).item()
        return action

    def get_fusion_details(self, obs_list):
        """
        Returns detailed fusion info for dashboard visualization.
        Args:
            obs_list (list[np.ndarray or torch.Tensor]): Features for each modality.
        Returns:
            dict: { 'raw_features': [...], 'fusion_weights': [...], 'fused_vector': [...], 'q_values': [...] }
        """
        features = [
            torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
            for x in obs_list
        ]
        details = {"raw_features": [f.cpu().numpy().tolist() for f in features]}
        fusion_type = getattr(self.model, "fusion_type", "concat")
        with torch.no_grad():
            if fusion_type == "concat":
                fused = torch.cat(features, dim=-1)
                details["fused_vector"] = fused.cpu().numpy().tolist()
                details["fusion_weights"] = None
            elif fusion_type == "attention":
                import torch.nn.functional as F

                weights = F.softmax(self.model.attn, dim=0).cpu().numpy().tolist()
                details["fusion_weights"] = weights
                padded = [
                    F.pad(f, (0, self.model.fusion_dim - f.shape[-1])) for f in features
                ]
                fused = sum(
                    w * p for w, p in zip(self.model.attn.softmax(dim=0), padded)
                )
                details["fused_vector"] = fused.cpu().numpy().tolist()
            elif fusion_type == "gating":
                x_cat = torch.cat(features, dim=-1)
                gates = self.model.gate(x_cat)  # shape: [batch, n_modalities]
                gates_np = gates.cpu().numpy().tolist()[0]
                details["fusion_weights"] = gates_np
                gated = [gates[:, i].unsqueeze(-1) * f for i, f in enumerate(features)]
                fused = torch.cat(gated, dim=-1)
                details["fused_vector"] = fused.cpu().numpy().tolist()
            else:
                details["fused_vector"] = None
                details["fusion_weights"] = None
            qvals = self.model(features)
            details["q_values"] = qvals.cpu().numpy().tolist()[0]
        return details

    def __repr__(self):
        return f"MultiModalFusionAgent(input_dims={self.model.input_dims})"
        # TODO: Add more informative metadata for model management
