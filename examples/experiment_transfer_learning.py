"""
experiment_transfer_learning.py

Sample experiment script for advanced transfer learning using CORAL alignment and MMD metric.
Compares baseline transfer learning (no adaptation) with CORAL-aligned transfer learning and prints MMD values.
"""

import numpy as np

from neuro_fuzzy_multiagent.environment.abstraction import SimpleEnvironment
from neuro_fuzzy_multiagent.environment.transfer_learning import (
    FeatureExtractor,
    coral,
    mmd,
    transfer_learning,
)


class DummyModel:
    """
    Minimal model for demonstration. Implements a forward method.
    """

    def forward(self, features):
        # Dummy operation: sum features
        return np.sum(features)


def collect_features(env, extractor, steps):
    features = []
    for _ in range(steps):
        state = env.reset()
        features.append(extractor.extract(state))
    return np.array(features)


def run_experiment():
    # Set random seed for reproducibility
    np.random.seed(0)
    # Create source and target environments
    src_env = SimpleEnvironment(dim=5)
    tgt_env = SimpleEnvironment(dim=5)
    # Feature extractor
    feat_extractor = FeatureExtractor(input_dim=5, output_dim=3)
    # Model
    model = DummyModel()
    steps = 20
    # Collect features before adaptation
    src_feats = collect_features(src_env, feat_extractor, steps)
    tgt_feats = collect_features(tgt_env, feat_extractor, steps)
    print("MMD (linear) BEFORE adaptation:", mmd(src_feats, tgt_feats, kernel="linear"))
    print(
        "MMD (rbf) BEFORE adaptation:",
        mmd(src_feats, tgt_feats, kernel="rbf", gamma=1.0),
    )
    print("=== Baseline Transfer Learning ===")
    transfer_learning(src_env, tgt_env, model, feat_extractor, steps=steps)
    print("Done.")
    print("\n=== Transfer Learning with CORAL Alignment ===")
    transfer_learning(
        src_env, tgt_env, model, feat_extractor, steps=steps, align_fn=coral
    )
    # Collect features after CORAL alignment
    src_feats_aligned = coral(src_feats, tgt_feats)
    print(
        "MMD (linear) AFTER CORAL:", mmd(src_feats_aligned, tgt_feats, kernel="linear")
    )
    print(
        "MMD (rbf) AFTER CORAL:",
        mmd(src_feats_aligned, tgt_feats, kernel="rbf", gamma=1.0),
    )
    print("Done.")


if __name__ == "__main__":
    run_experiment()
