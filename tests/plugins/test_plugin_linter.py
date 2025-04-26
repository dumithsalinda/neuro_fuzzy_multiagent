import tempfile
import os
from src.core.plugins.plugin_linter import PluginLinter


def test_lint_file_pass():
    code = "def foo():\n    return 42\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    linter = PluginLinter()
    passed, errors = linter.lint_file(fname)
    os.unlink(fname)
    assert passed
    assert errors == []


def test_lint_file_fail():
    code = "def bar():\n  return 43\n"  # Indentation error for flake8
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    linter = PluginLinter()
    passed, errors = linter.lint_file(fname)
    os.unlink(fname)
    assert not passed
    assert any("E" in e for e in errors) or errors  # Should have errors
