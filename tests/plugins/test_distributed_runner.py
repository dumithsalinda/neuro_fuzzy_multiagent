import time
from src.core.plugins.distributed_runner import DistributedTaskRunner


def test_run_distributed():
    runner = DistributedTaskRunner()

    def add(x, y):
        return x + y

    result = runner.run_distributed(add, 5, 7)
    assert result == 12


def test_map_distributed():
    runner = DistributedTaskRunner()

    def square(x):
        return x * x

    results = runner.map_distributed(square, [(2,), (3,), (4,)])
    assert results == [4, 9, 16]


def test_run_distributed_sleep():
    runner = DistributedTaskRunner()

    def slow(x):
        time.sleep(0.5)
        return x * 2

    result = runner.run_distributed(slow, 10)
    assert result == 20


def test_run_distributed_with_resources():
    runner = DistributedTaskRunner()

    def multiply(x, y):
        return x * y

    # This will run on any available node with CPU resource (simulate remote if cluster is set up)
    result = runner.run_distributed_with_resources(multiply, {"num_cpus": 1}, 3, 4)
    assert result == 12

    # Test error reporting from remote node
    def fail(x):
        raise RuntimeError("fail on remote")

    err = runner.run_distributed_with_resources(fail, {"num_cpus": 1}, 1)
    assert isinstance(err, Exception)
    assert "fail on remote" in str(err)
