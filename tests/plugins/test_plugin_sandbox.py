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


def plugin_sandbox_success(x):
    return x * 2


def plugin_sandbox_exception(x):
    raise ValueError("fail")


def plugin_sandbox_timeout(x):
    import time

    time.sleep(2)
    return x


def plugin_sandbox_cpu_limit():
    while True:
        pass


def plugin_sandbox_memory_limit():
    x = []
    while True:
        x.append(" " * 1024 * 1024)  # allocate 1MB per loop


def test_plugin_sandbox_success():
    sandbox = PluginSandbox()
    result = sandbox.run(plugin_sandbox_success, 3)
    assert result.success
    assert result.result == 6


def test_plugin_sandbox_exception():
    sandbox = PluginSandbox()
    result = sandbox.run(plugin_sandbox_exception, 3)
    assert not result.success
    assert "fail" in result.error


def test_plugin_sandbox_timeout():
    sandbox = PluginSandbox()
    result = sandbox.run(plugin_sandbox_timeout, 1, timeout=1)
    assert not result.success
    assert "timeout" in result.error.lower()


def test_plugin_sandbox_cpu_limit():
    sandbox = PluginSandbox()
    result = sandbox.run(plugin_sandbox_cpu_limit, timeout=5)
    assert not result.success
    assert (
        "cpu" in result.error.lower()
        or "killed" in result.error.lower()
        or "resource" in result.error.lower()
        or "signal" in result.error.lower()
    )


def test_plugin_sandbox_memory_limit():
    sandbox = PluginSandbox()
    result = sandbox.run(plugin_sandbox_memory_limit, timeout=5)
    assert not result.success
    assert (
        "memory" in result.error.lower()
        or "killed" in result.error.lower()
        or "resource" in result.error.lower()
        or "signal" in result.error.lower()
    )
