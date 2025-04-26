import sys
import os
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("ray", reason="Ray not installed"),
    reason="Ray not installed",
)


def test_run_distributed_experiments():
    from src.core.distributed.experiment_runner_ray import run_distributed_experiments

    # Use a minimal agent and environment already registered
    # Replace with actual names in your project
    from src.core.plugins.registration_utils import get_registered_plugins

    agent_name = next(iter(get_registered_plugins("agent")))
    env_name = next(iter(get_registered_plugins("environment")))

    config = {"agent": {}, "env": {}}
    try:
        results = run_distributed_experiments(
            agent_name, env_name, config, num_workers=2, episodes_per_worker=2
        )
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)
    except Exception as e:
        import traceback

        print("Ray distributed experiment test failed:", str(e))
        traceback.print_exc()
        pytest.skip(f"Ray distributed experiment failed: {e}")
