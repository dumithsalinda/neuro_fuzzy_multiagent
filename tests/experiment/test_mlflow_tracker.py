import os
import tempfile

from neuro_fuzzy_multiagent.core.experiment.mlflow_tracker import ExperimentTracker


def test_mlflow_experiment_tracker():
    # Use a temp directory for MLflow tracking URI
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["MLFLOW_TRACKING_URI"] = f"file://{tmpdir}"
        tracker = ExperimentTracker("test-exp")
        run_id = tracker.start_run(
            run_name="test-run", params={"foo": 42}, tags={"type": "unit-test"}
        )
        tracker.log_metrics({"acc": 0.95, "loss": 0.1})
        # Create a dummy artifact
        artifact_path = os.path.join(tmpdir, "dummy.txt")
        with open(artifact_path, "w") as f:
            f.write("artifact test")
        tracker.log_artifact(artifact_path)
        tracker.end_run()
        # Retrieve run and check
        run = tracker.get_run(run_id)
        assert run.data.params["foo"] == "42"
        assert run.data.metrics["acc"] == 0.95
        assert run.data.metrics["loss"] == 0.1
        assert run.info.status == "FINISHED"
