import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

pytestmark = pytest.mark.skipif(
    not pytest.importorskip('ray', reason='Ray not installed'),
    reason="Ray not installed"
)

def test_run_distributed_experiments():
    from core.distributed.experiment_runner_ray import run_distributed_experiments
    # Use a minimal agent and environment already registered
    # Replace with actual names in your project
    agent_name = next(iter(__import__('core.plugins.registration_utils').plugins['agent']))
    env_name = next(iter(__import__('core.plugins.registration_utils').plugins['environment']))
    config = {'agent': {}, 'env': {}}
    results = run_distributed_experiments(agent_name, env_name, config, num_workers=2, episodes_per_worker=2)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, list) for r in results)
