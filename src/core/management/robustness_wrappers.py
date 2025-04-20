"""
robustness_wrappers.py

Wrappers for adding noise, adversarial perturbations, and safety checks to multi-agent environments or agents.
Ensures original (core) fuzzy rules are never overwritten.
"""

import numpy as np


class ObservationNoiseWrapper:
    """
    Adds Gaussian noise to observations.
    """

    def __init__(self, env, noise_std=0.1):
        self.env = env
        self.noise_std = noise_std

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        noisy_obs = [
            o + np.random.normal(0, self.noise_std, size=np.shape(o)) for o in obs
        ]
        return noisy_obs, reward, done, info


class ActionPerturbationWrapper:
    """
    Adds random or adversarial noise to agent actions.
    """

    def __init__(self, env, perturb_prob=0.1, n_actions=None):
        self.env = env
        self.perturb_prob = perturb_prob
        self.n_actions = n_actions

    def reset(self):
        return self.env.reset()

    def step(self, action):
        perturbed_action = action
        if np.random.rand() < self.perturb_prob:
            if self.n_actions is not None:
                perturbed_action = np.random.randint(self.n_actions)
            else:
                perturbed_action = action  # fallback: no perturbation
        return self.env.step(perturbed_action)


class SafetyMonitor:
    """
    Monitors agent actions for safety constraint violations.
    """

    def __init__(self, constraints=None):
        self.constraints = (
            constraints or []
        )  # List of callables: f(obs, action) -> bool
        self.violations = []

    def check(self, obs, action):
        for constraint in self.constraints:
            if not constraint(obs, action):
                self.violations.append((obs, action))
                print(f"[SafetyMonitor] Violation detected: obs={obs}, action={action}")
                return False
        return True
