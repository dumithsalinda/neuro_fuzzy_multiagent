import numpy as np
from src.core.agents.hpo import HyperparameterOptimizer


def test_hpo_finds_best():
    # True optimum at x=3, y=5
    def eval_fn(params):
        x, y = params["x"], params["y"]
        return -((x - 3) ** 2 + (y - 5) ** 2)

    param_space = {"x": np.arange(0, 7), "y": np.arange(0, 11)}
    hpo = HyperparameterOptimizer(param_space, eval_fn)
    best_params, best_score = hpo.optimize(n_trials=100)
    assert abs(best_params["x"] - 3) <= 1
    assert abs(best_params["y"] - 5) <= 1
    assert best_score >= -2


def test_hpo_history():
    def eval_fn(params):
        return params["a"]

    param_space = {"a": [1, 2, 3]}
    hpo = HyperparameterOptimizer(param_space, eval_fn)
    hpo.optimize(n_trials=5)
    hist = hpo.get_history()
    assert len(hist) == 5
    assert all("a" in p for p, _ in hist)
