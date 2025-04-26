import os
import shutil
import sys
import tempfile

import pytest

# Add src directory to sys.path for src-layout compatibility
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.insert(0, src_path)
from neuro_fuzzy_multiagent.core.experiment.result_analysis import ResultAnalyzer


def test_generate_report_and_save(tmp_path):
    analyzer = ResultAnalyzer(output_dir=tmp_path)
    config = {"lr": 0.01, "epochs": 10}
    metrics = {"accuracy": 0.95, "loss": 0.1}
    report = analyzer.generate_report(config, metrics, run_id="test123")
    assert "Experiment Report" in report
    path = analyzer.save_report(report, filename="test_report.md")
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    assert "accuracy" in content
    assert "test123" in content


def test_export_metrics_json_and_csv(tmp_path):
    analyzer = ResultAnalyzer(output_dir=tmp_path)
    metrics = {"accuracy": 0.9, "loss": 0.2}
    json_path = analyzer.export_metrics(metrics, filename="metrics.json")
    assert os.path.exists(json_path)
    import json

    with open(json_path) as f:
        data = json.load(f)
    assert data["accuracy"] == 0.9
    csv_path = analyzer.export_metrics(metrics, filename="metrics.csv")
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        content = f.read()
    assert "accuracy" in content
    assert "loss" in content


def test_summary_statistics():
    analyzer = ResultAnalyzer()
    metric_history = {"acc": [0.8, 0.85, 0.9], "loss": [0.2, 0.15, 0.1]}
    stats = analyzer.summary_statistics(metric_history)
    assert "acc" in stats
    assert "mean" in stats["acc"]
    assert abs(stats["acc"]["mean"] - 0.85) < 1e-6
    assert "loss" in stats
    assert stats["loss"]["min"] == 0.1


def test_plot_metrics(tmp_path):
    analyzer = ResultAnalyzer(output_dir=tmp_path)
    metric_history = {"acc": [0.7, 0.8, 0.9], "loss": [0.3, 0.2, 0.1]}
    try:
        path = analyzer.plot_metrics(metric_history, filename="plot.png")
        if path:
            assert os.path.exists(path)
    except ImportError:
        pytest.skip("matplotlib not installed")
