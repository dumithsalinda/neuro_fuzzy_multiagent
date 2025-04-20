import os
import json
import datetime
import uuid


class ExperimentManager:
    """
    Manages experiment tracking, versioning, and result aggregation for agent experiments.
    Stores metadata, parameters, and results for each run.
    """

    def __init__(self, log_dir="experiments"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.runs = []

    def start_run(self, params: dict):
        run_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        run_info = {
            "run_id": run_id,
            "timestamp": timestamp,
            "params": params,
            "results": None,
            "log_file": os.path.join(self.log_dir, f"run_{run_id}.json"),
        }
        self.runs.append(run_info)
        return run_info

    def log_results(self, run_info: dict, results: dict):
        run_info["results"] = results
        with open(run_info["log_file"], "w") as f:
            json.dump(
                {
                    "params": run_info["params"],
                    "results": results,
                    "timestamp": run_info["timestamp"],
                },
                f,
                indent=2,
            )

    def aggregate_results(self, metric: str):
        # Aggregate a metric across all runs
        values = []
        for run in self.runs:
            if run["results"] and metric in run["results"]:
                values.append(run["results"][metric])
        return values

    def list_runs(self):
        return self.runs

    def load_all_runs(self):
        # Load all run logs from disk
        for fname in os.listdir(self.log_dir):
            if fname.endswith(".json"):
                with open(os.path.join(self.log_dir, fname), "r") as f:
                    data = json.load(f)
                    self.runs.append(
                        {
                            "run_id": fname.split("_")[1].split(".")[0],
                            "timestamp": data.get("timestamp"),
                            "params": data.get("params"),
                            "results": data.get("results"),
                            "log_file": os.path.join(self.log_dir, fname),
                        }
                    )


# Usage Example:
# mgr = ExperimentManager()
# run = mgr.start_run({"agent_type": "DQNAgent", "env": "Gridworld"})
# ... run experiment ...
# mgr.log_results(run, {"mean_reward": 10.2, "std_reward": 2.3})
# all_means = mgr.aggregate_results("mean_reward")
