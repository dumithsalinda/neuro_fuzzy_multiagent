import tempfile
import os
from pathlib import Path
from src.core.plugins.plugin_dependency import PluginDependencyManager


def test_no_requirements():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = PluginDependencyManager(tmpdir)
        assert not mgr.has_requirements()
        ok, msg = mgr.check_requirements_installed()
        assert ok
        assert "No requirements" in msg


def test_fake_requirements():
    with tempfile.TemporaryDirectory() as tmpdir:
        req_path = Path(tmpdir) / "requirements.txt"
        req_path.write_text("nonexistentpackage1234567890\n")
        mgr = PluginDependencyManager(tmpdir)
        assert mgr.has_requirements()
        ok, msg = mgr.check_requirements_installed()
        assert not ok
        assert "nonexistentpackage1234567890" in msg
        # Don't actually install


def test_real_requirements():
    with tempfile.TemporaryDirectory() as tmpdir:
        req_path = Path(tmpdir) / "requirements.txt"
        req_path.write_text("pytest\n")
        mgr = PluginDependencyManager(tmpdir)
        assert mgr.has_requirements()
        ok, msg = mgr.check_requirements_installed()
        assert ok or "Would install" in msg or "already satisfied" in msg
        # Don't actually install
