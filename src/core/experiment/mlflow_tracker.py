import mlflow
from typing import Dict, Any

class ExperimentTracker:
    """
    Wrapper for MLflow experiment tracking.
    """
    def __init__(self, experiment_name="default"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name=None, params: Dict[str, Any]=None, tags: Dict[str, str]=None):
        self.run = mlflow.start_run(run_name=run_name)
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)
        return self.run.info.run_id

    def log_metrics(self, metrics: Dict[str, float]):
        mlflow.log_metrics(metrics)

    def log_artifact(self, file_path):
        mlflow.log_artifact(file_path)

    def end_run(self, status="FINISHED"):
        mlflow.end_run(status=status)

    def get_run(self, run_id):
        return mlflow.get_run(run_id)

# Example usage (for test):
if __name__ == "__main__":
    tracker = ExperimentTracker("demo-exp")
    run_id = tracker.start_run(run_name="test", params={"foo": 1})
    tracker.log_metrics({"acc": 0.99})
    tracker.end_run()
    print(tracker.get_run(run_id))
