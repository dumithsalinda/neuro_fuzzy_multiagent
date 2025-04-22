from src.core.plugins.plugin_sandbox import PluginSandbox
import time

def plugin_success(x):
    return x + 1

def plugin_exception(x):
    raise ValueError("fail")

def plugin_timeout(x):
    import time
    time.sleep(5)
    return x

from src.core.plugins.plugin_sandbox import PluginSandbox

def test_plugin_success():
    sandbox = PluginSandbox(timeout=2)
    result = sandbox.run(plugin_success, 41)
    assert result.success
    assert result.result == 42

def test_plugin_exception():
    sandbox = PluginSandbox(timeout=2)
    result = sandbox.run(plugin_exception, 1)
    assert not result.success
    assert "fail" in result.error

def test_plugin_timeout():
    sandbox = PluginSandbox(timeout=1)
    result = sandbox.run(plugin_timeout, 1)
    assert not result.success
    assert result.error == "Timeout"
