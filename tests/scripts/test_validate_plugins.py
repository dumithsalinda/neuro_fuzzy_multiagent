import sys
import subprocess
import pytest

@pytest.mark.skipif(sys.version_info[0] < 3, reason="Requires Python 3+")
def test_validate_plugins_cli():
    proc = subprocess.Popen([
        sys.executable, "scripts/validate_plugins.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = proc.communicate()
    out = out.decode("utf-8") if hasattr(out, 'decode') else out
    assert proc.returncode in (0, 1)
    assert "Validating plugin interfaces" in out
    if proc.returncode == 1:
        assert "[FAIL]" in out
    else:
        assert "All plugins passed validation." in out
