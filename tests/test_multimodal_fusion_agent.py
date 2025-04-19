"""
test_multimodal_fusion_agent.py

Test for MultiModalFusionAgent with synthetic multi-modal data (image + text).
"""
import numpy as np
import torch
from src.core.multimodal_fusion_agent import MultiModalFusionAgent

def test_multimodal_fusion_agent_forward():
    # Two modalities: image (4-dim), text (3-dim)
    img_dim, txt_dim, n_actions = 4, 3, 5
    agent = MultiModalFusionAgent(
        input_dims=[img_dim, txt_dim],
        hidden_dim=16,
        output_dim=n_actions,
        fusion_type='concat',
    )
    # Batch of 2 samples
    img_feats = np.random.randn(2, img_dim)
    txt_feats = np.random.randn(2, txt_dim)
    img_tensor = torch.tensor(img_feats, dtype=torch.float32)
    txt_tensor = torch.tensor(txt_feats, dtype=torch.float32)
    qvals = agent.model([img_tensor, txt_tensor])  # shape: [2, n_actions]
    assert qvals.shape == (2, n_actions)
    # Check that output is finite
    assert torch.isfinite(qvals).all()
    print("Q-values:", qvals)

def test_multimodal_fusion_agent_q_update():
    # Single step Q-learning update
    img_dim, txt_dim, n_actions = 4, 3, 5
    agent = MultiModalFusionAgent(
        input_dims=[img_dim, txt_dim],
        hidden_dim=16,
        output_dim=n_actions,
        fusion_type='concat',
    )
    obs = [np.random.randn(img_dim), np.random.randn(txt_dim)]
    next_obs = [np.random.randn(img_dim), np.random.randn(txt_dim)]
    action = 2
    reward = 1.0
    done = False
    agent.observe(obs, action, reward, next_obs, done)
    # Loss history should be updated
    assert hasattr(agent, "loss_history") and len(agent.loss_history) > 0
    print("Loss history:", list(agent.loss_history))
