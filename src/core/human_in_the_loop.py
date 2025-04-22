from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Optional
import threading

app = FastAPI()

class ActionApproval(BaseModel):
    observation: Any
    proposed_action: Any
    approved: Optional[bool] = None
    modified_action: Optional[Any] = None

# Shared state for experiment loop to wait for human input
action_event = threading.Event()
approval_data = {}

def request_human_approval(observation, proposed_action):
    global approval_data
    approval_data = {
        "observation": observation,
        "proposed_action": proposed_action,
        "approved": None,
        "modified_action": None
    }
    action_event.clear()
    action_event.wait()  # Block until human responds
    return approval_data["approved"], approval_data["modified_action"]

@app.post("/human/approve_action")
async def approve_action(approval: ActionApproval):
    global approval_data
    approval_data["approved"] = approval.approved
    approval_data["modified_action"] = approval.modified_action
    action_event.set()
    return JSONResponse({"status": "received"})

@app.get("/human/pending_action")
async def pending_action():
    if approval_data:
        return approval_data
    return {"status": "no pending action"}
