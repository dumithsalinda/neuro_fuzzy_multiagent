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
        result = plugin_callable(*args, **kwargs)
        result_queue.put(PluginSandboxResult(True, result=result))
    except Exception as e:
        tb = traceback.format_exc()
        result_queue.put(PluginSandboxResult(False, error=str(e), traceback_str=tb))

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
            return result_queue.get()
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
