from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()


@app.get("/data")
def get_data():
    # Simulate a sensor or robot state
    return {"temperature": 23.5, "status": "OK", "position": [1.0, 2.0]}


@app.post("/data")
def post_data(request: Request):
    # Simulate accepting an action/command
    import asyncio

    data = asyncio.run(request.json())
    # Echo back the action with a mock result
    return {"received_action": data, "result": "executed", "status": "OK"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
