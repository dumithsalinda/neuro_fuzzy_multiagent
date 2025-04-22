"""
Audit logging for human-in-the-loop approvals/denials.
Logs each decision with timestamp, agent, action, and context.
"""
import os
import csv
from datetime import datetime

AUDIT_LOG_PATH = os.environ.get("NFMA_HUMAN_AUDIT_LOG", "human_approval_audit.csv")

LOG_FIELDS = ["timestamp", "agent", "action", "decision", "context"]

def log_human_decision(agent_name, action, decision, context=None):
    """
    Log a human approval/denial event for auditability.
    decision: 'approved' or 'denied'
    context: (optional) dict with extra info
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "agent": agent_name,
        "action": action,
        "decision": decision,
        "context": str(context) if context else ""
    }
    file_exists = os.path.isfile(AUDIT_LOG_PATH)
    with open(AUDIT_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)

def clear_audit_log():
    if os.path.exists(AUDIT_LOG_PATH):
        os.remove(AUDIT_LOG_PATH)

def read_audit_log():
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    with open(AUDIT_LOG_PATH, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)
