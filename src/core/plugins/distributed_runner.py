import ray
import time


class DistributedTaskRunner:
    """
    Provides a simple Ray-based distributed task runner for agents/environments.

    Features:
    - Call .run_distributed(func, *args, **kwargs) to execute a function in Ray.
    - Use .run_distributed_with_resources(func, resource_dict, *args, **kwargs) to target a specific node (remote or local) with Ray resources.
    - All exceptions/errors from remote tasks are captured and returned as Python exceptions.

    Example (remote execution):
        runner = DistributedTaskRunner()
        # This will run on a node with the custom resource 'remote_node:1', if available
        def plugin_task(x):
            return x * 10
        result = runner.run_distributed_with_resources(plugin_task, {"remote_node": 1}, 5)
        print(result)  # 50
    """

    def __init__(self, address=None):
        if not ray.is_initialized():
            ray.init(address=address, ignore_reinit_error=True, log_to_driver=False)

    @staticmethod
    @ray.remote
    def _remote_func(func, *args, **kwargs):
        return func(*args, **kwargs)

    def run_distributed(self, func, *args, **kwargs):
        """
        Runs the function as a Ray remote task and returns the result.
        """
        future = self._remote_func.remote(func, *args, **kwargs)
        return ray.get(future)

    def map_distributed(self, func, args_list):
        """
        Runs func with each args tuple in args_list as Ray remote tasks.
        Returns list of results.
        """
        futures = [self._remote_func.remote(func, *args) for args in args_list]
        return ray.get(futures)

    def run_distributed_with_resources(self, func, resource_kwargs, *args, **kwargs):
        """
        Runs the function as a Ray remote task on a node with the specified resources.
        Example: resource_kwargs={"num_cpus": 1} or {"resources": {"remote_node": 1}}
        Returns result or raises the remote exception.
        """
        remote_func = ray.remote(**resource_kwargs)(func)
        future = remote_func.remote(*args, **kwargs)
        try:
            return ray.get(future)
        except Exception as e:
            return e


# Example usage (for test):
if __name__ == "__main__":
    runner = DistributedTaskRunner()

    def slow_add(x, y):
        time.sleep(1)
        return x + y

    print(runner.run_distributed(slow_add, 2, 3))
    print(runner.map_distributed(slow_add, [(1, 2), (3, 4)]))
