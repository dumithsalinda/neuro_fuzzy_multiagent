import mlflow
from typing import Dict, Any, Optional
import logging


class ExperimentTracker:
    """
    Wrapper for MLflow experiment tracking.
    Provides methods to start runs, log parameters/metrics/artifacts, and retrieve run info.
    """

    def __init__(self, experiment_name: str = "default"):
        """
        Args:
            experiment_name (str): Name of the MLflow experiment.
        """
        self.experiment_name: str = experiment_name
        try:
            mlflow.set_experiment(experiment_name)
            logging.info(f"MLflow experiment set: {experiment_name}")
        except Exception as e:
            logging.error(f"Failed to set MLflow experiment: {e}")
        self.run = None

    def start_run(
        self,
        run_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Start a new MLflow run and optionally log parameters and tags.
        Returns the run ID.
        """
        try:
            self.run = mlflow.start_run(run_name=run_name)
            if params:
                mlflow.log_params(params)
            if tags:
                mlflow.set_tags(tags)
            logging.info(f"Started MLflow run: {self.run.info.run_id}")
            return self.run.info.run_id
        except Exception as e:
            logging.error(f"Failed to start MLflow run: {e}")
            return None

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log metrics to the current MLflow run.
        """
        try:
            mlflow.log_metrics(metrics)
            logging.info(f"Logged metrics: {metrics}")
        except Exception as e:
            logging.error(f"Failed to log metrics: {e}")

    def log_artifact(self, file_path: str) -> None:
        """
        Log an artifact (file) to the current MLflow run.
        """
        try:
            mlflow.log_artifact(file_path)
            logging.info(f"Logged artifact: {file_path}")
        except Exception as e:
            logging.error(f"Failed to log artifact: {e}")

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.
        """
        try:
            mlflow.end_run(status=status)
            logging.info(f"Ended MLflow run with status: {status}")
        except Exception as e:
            logging.error(f"Failed to end MLflow run: {e}")

    def get_run(self, run_id: str) -> Optional[Any]:
        """
        Retrieve information about a specific MLflow run.
        """
        try:
            run = mlflow.get_run(run_id)
            logging.info(f"Retrieved MLflow run: {run_id}")
            return run
        except Exception as e:
            logging.error(f"Failed to retrieve MLflow run {run_id}: {e}")
            return None


# Example usage (for test):
if __name__ == "__main__":
    tracker = ExperimentTracker("demo-exp")
    run_id = tracker.start_run(run_name="test", params={"foo": 1})
    tracker.log_metrics({"acc": 0.99})
    tracker.end_run()
    print(tracker.get_run(run_id))
