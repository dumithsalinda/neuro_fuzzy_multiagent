import numpy as np
import torch
import torch.nn as nn
from src.core.neural_networks.fusion import FusionNetwork
from src.core.neural_networks.fuzzy_system import FuzzyInferenceSystem


class NeuroFuzzyFusionAgent:
    """
    Multi-modal agent combining neural and fuzzy logic for decision making.
    Accepts multiple input modalities, fuses both neural and fuzzy outputs.
    """

    def __init__(
        self,
        input_dims,
        hidden_dim,
        output_dim,
        fusion_type="concat",
        fuzzy_config=None,
        fusion_alpha=0.5,
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Neural fusion network
        self.neural_fusion = FusionNetwork(
            input_dims, hidden_dim, output_dim, fusion_type=fusion_type
        ).to(self.device)
        # Fuzzy inference system
        self.fuzzy_system = (
            FuzzyInferenceSystem(fuzzy_config)
            if fuzzy_config is not None
            else FuzzyInferenceSystem()
        )
        # Fusion parameter (0=all fuzzy, 1=all neural)
        self.fusion_alpha = fusion_alpha
        self.output_dim = output_dim
        self.optimizer = torch.optim.Adam(self.neural_fusion.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.train_mode = False
        self.neural_fusion.eval()

    def act(self, obs_list):
        """
        Args:
            obs_list: list of modality features (numpy arrays or tensors)
        Returns:
            int: selected action
        """
        # Neural output
        features = [
            torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
            for x in obs_list
        ]
        with torch.no_grad():
            neural_out = self.neural_fusion(features).cpu().numpy().flatten()
        # Fuzzy output
        # Concatenate all modality features into a single flat vector for fuzzy inference
        obs_vector = np.concatenate([np.asarray(x).flatten() for x in obs_list])
        fuzzy_out = self.fuzzy_system.evaluate(
            obs_vector
        )  # Returns a scalar or vector depending on rules
        # If output is scalar, convert to vector for fusion
        if np.isscalar(fuzzy_out):
            fuzzy_out = np.full(self.output_dim, fuzzy_out)
        # Fuse
        fused_out = self.fusion_alpha * neural_out + (1 - self.fusion_alpha) * np.array(
            fuzzy_out
        )
        action = int(np.argmax(fused_out))
        return action

    def explain_action(self, obs_list):
        """
        Returns a dict explaining both neural and fuzzy reasoning.
        """
        features = [
            torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
            for x in obs_list
        ]
        with torch.no_grad():
            neural_out = self.neural_fusion(features).cpu().numpy().flatten()
        fuzzy_exp = self.fuzzy_system.explain_inference(obs_list)
        fuzzy_out = fuzzy_exp.get("output", None)
        fused_out = self.fusion_alpha * neural_out + (1 - self.fusion_alpha) * np.array(
            fuzzy_out
        )
        return {
            "neural_output": neural_out.tolist(),
            "fuzzy_explanation": fuzzy_exp,
            "fused_output": fused_out.tolist(),
            "chosen_action": int(np.argmax(fused_out)),
        }

    def train_step(self, obs_list, target):
        """
        One training step for the neural fusion part.
        Args:
            obs_list: list of modality features
            target: target Q-values or policy vector
        """
        self.neural_fusion.train()
        features = [
            torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
            for x in obs_list
        ]
        target = torch.tensor(
            target, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        output = self.neural_fusion(features)
        loss = self.loss_fn(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.neural_fusion.eval()
        return loss.item()

    def set_fusion_alpha(self, alpha):
        """Set the neural/fuzzy fusion ratio (0=fuzzy only, 1=neural only)."""
        self.fusion_alpha = float(alpha)

    def add_fuzzy_rule_from_feedback(
        self, antecedent_values, consequent, fuzzy_sets_per_input
    ):
        """
        Add a fuzzy rule based on human feedback.
        antecedent_values: list of floats (input values for the rule antecedents)
        consequent: float or int (desired output/action)
        fuzzy_sets_per_input: list of lists of FuzzySet for each input feature
        """
        self.fuzzy_system.add_rule_from_feedback(
            antecedent_values, consequent, fuzzy_sets_per_input
        )
