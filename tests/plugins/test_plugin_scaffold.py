import subprocess
import os
import tempfile


def test_scaffold_environment_plugin():
    with tempfile.TemporaryDirectory() as tmpdir:
        script = os.path.abspath("plugin_scaffold.py")
        env_dir = os.path.join(tmpdir, "src/env/")
        os.makedirs(env_dir, exist_ok=True)
        # Run CLI to create plugin
        result = subprocess.run(
            ["python3", script, "environment", "TestEnvPlugin"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        expected_file = os.path.join(env_dir, "testenvplugin.py")
        assert os.path.exists(expected_file)
        with open(expected_file) as f:
            content = f.read()
            assert "class TestEnvPlugin" in content
            assert "BaseEnvironment" in content
