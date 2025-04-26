import os
from typing import Dict, Any, Optional
from datetime import datetime

import logging
import json
import csv
from statistics import mean, stdev
from typing import Dict, Any, Optional, List
from datetime import datetime


class ResultAnalyzer:
    """
    Utility to generate experiment result summary reports, plots, exports, and statistics.
    """

    def __init__(self, output_dir: str = "."):
        """
        Args:
            output_dir (str): Directory to save reports and artifacts.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_report(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        run_id: Optional[str] = None,
    ) -> str:
        """
        Generate a Markdown report summarizing config and metrics.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"""# Experiment Report\n\n"""
        if run_id:
            report += f"**MLflow Run ID:** `{run_id}`  \n"
        report += f"**Generated:** {now}\n\n"
        report += "## Configuration\n"
        for k, v in config.items():
            report += f"- **{k}**: {v}\n"
        report += "\n## Metrics\n"
        for k, v in metrics.items():
            report += f"- **{k}**: {v}\n"
        return report

    def save_report(self, report_md: str, filename: str = "report.md") -> str:
        """
        Save a Markdown report to file.
        """
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write(report_md)
        logging.info(f"Saved report to {path}")
        return path

    def plot_metrics(
        self, metric_history: Dict[str, List[float]], filename: str = "metrics.png"
    ) -> Optional[str]:
        """
        Plot metric histories as line plots and save as an image (requires matplotlib).
        Args:
            metric_history (dict): Dict of metric name -> list of values.
            filename (str): Output image filename.
        Returns:
            Path to saved plot or None if matplotlib not available.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logging.warning("matplotlib not installed, skipping plot_metrics.")
            return None
        plt.figure()
        for k, v in metric_history.items():
            plt.plot(v, label=k)
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.title("Experiment Metrics")
        plt.legend()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path)
        plt.close()
        logging.info(f"Saved metrics plot to {path}")
        return path

    def export_metrics(
        self, metrics: Dict[str, Any], filename: str = "metrics.json"
    ) -> str:
        """
        Export metrics to a JSON or CSV file.
        Args:
            metrics (dict): Metrics to export.
            filename (str): Output file name (.json or .csv).
        Returns:
            Path to saved file.
        """
        path = os.path.join(self.output_dir, filename)
        if filename.endswith(".json"):
            with open(path, "w") as f:
                json.dump(metrics, f, indent=2)
        elif filename.endswith(".csv"):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for k, v in metrics.items():
                    writer.writerow([k, v])
        else:
            raise ValueError("Unsupported export file format.")
        logging.info(f"Exported metrics to {path}")
        return path

    def summary_statistics(
        self, metric_history: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics (mean, std, min, max) for each metric.
        Args:
            metric_history (dict): Dict of metric name -> list of float values.
        Returns:
            Dict of metric name -> dict of statistics.
        """
        stats = {}
        for k, v in metric_history.items():
            if not v:
                continue
            stats[k] = {
                "mean": mean(v),
                "std": stdev(v) if len(v) > 1 else 0.0,
                "min": min(v),
                "max": max(v),
            }
        logging.info(f"Computed summary statistics: {stats}")
        return stats
