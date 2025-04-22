"""
plugin_submission.py

Local storage for plugin submission requests (JSON file).
"""
import json
import os
from datetime import datetime

SUBMISSIONS_PATH = "pending_plugin_submissions.json"


def load_submissions():
    if not os.path.exists(SUBMISSIONS_PATH):
        return []
    with open(SUBMISSIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_submissions(data):
    with open(SUBMISSIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def add_submission(submission):
    data = load_submissions()
    submission["timestamp"] = datetime.utcnow().isoformat()
    data.append(submission)
    save_submissions(data)

# --- Remote API Integration ---
REMOTE_API_URL = None  # Set to your API endpoint, e.g. 'https://my-marketplace.com/api/submit'

def post_submission_to_remote(submission):
    """
    POST the submission to the remote API endpoint.
    Returns (success: bool, message: str)
    """
    if not REMOTE_API_URL:
        return False, "No remote API endpoint configured."
    try:
        import requests
        resp = requests.post(REMOTE_API_URL, json=submission, timeout=10)
        if resp.status_code == 200:
            return True, "Submission posted to remote API."
        else:
            return False, f"Remote API error: {resp.status_code} {resp.text}"
    except Exception as e:
        return False, f"Failed to post to remote API: {e}"
