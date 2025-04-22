import multiprocessing
import traceback
import sys

class PluginSandboxResult:
    def __init__(self, success, result=None, error=None, traceback_str=None):
        self.success = success
        self.result = result
        self.error = error
        self.traceback = traceback_str

def _sandbox_runner(plugin_callable, args, kwargs, result_queue):
    try:
        # Set resource limits (Unix only)
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
            mem_bytes = 256 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except Exception:
            pass  # Not available on all systems
        result = plugin_callable(*args, **kwargs)
        # Only pass built-in types (tuple) through the queue
        result_queue.put((True, result, None, None))
    except Exception as e:
        tb = traceback.format_exc()
        result_queue.put((False, None, str(e), tb))

class PluginSandbox:
    def __init__(self, timeout=10):
        self.timeout = timeout

    def run(self, plugin_callable, *args, **kwargs):
        """
        Runs the given plugin_callable in a subprocess with a timeout.
        Returns PluginSandboxResult.
        """
        ctx = multiprocessing.get_context('spawn')
        result_queue = ctx.Queue()
        p = ctx.Process(target=_sandbox_runner, args=(plugin_callable, args, kwargs, result_queue))
        p.start()
        p.join(self.timeout)
        if p.is_alive():
            p.terminate()
            return PluginSandboxResult(False, error="Timeout", traceback_str=None)
        if not result_queue.empty():
            tup = result_queue.get()
            # tup: (success, result, error, traceback)
            return PluginSandboxResult(*tup)
        return PluginSandboxResult(False, error="No result returned", traceback_str=None)

# Example usage (for test):
if __name__ == "__main__":
    def test_plugin(x):
        return x * 2
    sandbox = PluginSandbox(timeout=2)
    result = sandbox.run(test_plugin, 5)
    print("Success:", result.success)
    print("Result:", result.result)
    print("Error:", result.error)
    print("Traceback:", result.traceback)
