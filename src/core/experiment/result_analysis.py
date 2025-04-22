import os
from typing import Dict, Any, Optional
from datetime import datetime

class ResultAnalyzer:
    """
    Utility to generate experiment result summary reports and plots, and log as artifacts.
    """
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_report(self, config: Dict[str, Any], metrics: Dict[str, Any], run_id: Optional[str] = None) -> str:
        """Generate a Markdown report summarizing config and metrics."""
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
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write(report_md)
        return path

    # Optionally, add plot generation (if metric history is available)
