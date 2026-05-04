"""Report generation: renders the FrameworkReport to HTML/PDF (Phase 6)."""

from __future__ import annotations

from pathlib import Path

from .utils import FrameworkReport


def generate_report(report: FrameworkReport, output_dir: str | Path, config: dict) -> Path:
    """Render *report* to an HTML file inside *output_dir*.

    Args:
        report: Fully populated FrameworkReport.
        output_dir: Directory where the report file will be written.
        config: Reporting configuration block from the YAML file.

    Returns:
        Path to the generated report file.
    """
    raise NotImplementedError("Report generation will be implemented in Phase 6.")
