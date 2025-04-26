from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Any, Optional, Tuple
import threading
import logging

app = FastAPI(
    title="Human-in-the-Loop API",
    description="APIs for human approval of agent actions.",
)


class ActionApproval(BaseModel):
    observation: Any = Field(..., description="Observation presented to the agent.")
    proposed_action: Any = Field(..., description="Action proposed by the agent.")
    approved: Optional[bool] = Field(
        None, description="Whether the action is approved by the human."
    )
    modified_action: Optional[Any] = Field(
        None, description="Modified action if the human changes it."
    )


# Shared state for experiment loop to wait for human input
action_event = threading.Event()
approval_data = {}


def request_human_approval(
    observation: Any, proposed_action: Any
) -> Tuple[Optional[bool], Optional[Any]]:
    """
    Block the experiment loop until human approval is received for the given action.
    Returns (approved, modified_action).
    """
    global approval_data
    approval_data = {
        "observation": observation,
        "proposed_action": proposed_action,
        "approved": None,
        "modified_action": None,
    }
    action_event.clear()
    logging.info(
        f"Waiting for human approval: observation={observation}, proposed_action={proposed_action}"
    )
    action_event.wait()  # Block until human responds
    logging.info(
        f"Human approval received: approved={approval_data['approved']}, modified_action={approval_data['modified_action']}"
    )
    return approval_data["approved"], approval_data["modified_action"]


@app.post("/human/approve_action", tags=["Human-in-the-Loop"])
async def approve_action(approval: ActionApproval) -> JSONResponse:
    """
    Endpoint for human to approve or modify an agent's action.
    """
    global approval_data
    try:
        approval_data["approved"] = approval.approved
        approval_data["modified_action"] = approval.modified_action
        action_event.set()
        logging.info(f"Approval data updated: {approval_data}")
        return JSONResponse({"status": "received"})
    except Exception as e:
        logging.error(f"Error in approve_action: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/human/pending_action", tags=["Human-in-the-Loop"])
async def pending_action() -> dict:
    """
    Get the current pending action awaiting human approval.
    """
    if approval_data:
        logging.info(f"Pending approval data: {approval_data}")
        return approval_data
    return {"status": "no pending action"}
