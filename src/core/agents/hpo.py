import numpy as np
from typing import Callable, Dict, Any, List


class HyperparameterOptimizer:
    """
    Simple Bayesian Optimization/Evolutionary Strategy for agent hyperparameter tuning.
    Usage:
        optimizer = HyperparameterOptimizer(param_space, eval_fn)
        best_params, best_score = optimizer.optimize(n_trials=20)
    """

    def __init__(
        self,
        param_space: Dict[str, List[Any]],
        eval_fn: Callable[[Dict[str, Any]], float],
    ):
        self.param_space = param_space
        self.eval_fn = eval_fn
        self.history = []  # List of (params, score)

    def sample_params(self) -> Dict[str, Any]:
        return {k: np.random.choice(v) for k, v in self.param_space.items()}

    def optimize(self, n_trials=20) -> (Dict[str, Any], float):
        best_params = None
        best_score = -np.inf
        for _ in range(n_trials):
            params = self.sample_params()
            score = self.eval_fn(params)
            self.history.append((params, score))
            if score > best_score:
                best_params = params
                best_score = score
        return best_params, best_score

    def get_history(self):
        return self.history
