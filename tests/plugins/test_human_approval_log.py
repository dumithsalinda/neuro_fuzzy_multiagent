import sys
import os
import tempfile

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)
from src.core.plugins.human_approval_log import (
    log_human_decision,
    read_audit_log,
    clear_audit_log,
)


def test_log_and_read_audit(tmp_path):
    # Patch the log path
    audit_path = tmp_path / "audit.csv"
    os.environ["NFMA_HUMAN_AUDIT_LOG"] = str(audit_path)
    clear_audit_log()
    log_human_decision("TestAgent", "move", "approved", {"obs": 42})
    log_human_decision("TestAgent", "move", "denied", {"obs": 99})
    rows = read_audit_log()
    assert len(rows) == 2
    assert rows[0]["decision"] == "approved"
    assert rows[1]["decision"] == "denied"
    assert rows[0]["agent"] == "TestAgent"
    assert rows[0]["action"] == "move"
    assert "obs" in rows[0]["context"]
    clear_audit_log()
    assert read_audit_log() == []
