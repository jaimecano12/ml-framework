"""Report generation — renders a FrameworkReport to HTML (Phase 6)."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from .utils import CheckResult, FrameworkReport, logger

matplotlib.use("Agg")  # non-interactive backend — safe in all environments

_TEMPLATES_DIR = Path(__file__).parent / "templates"


# ---------------------------------------------------------------------------
# Plot helpers (all return base64-encoded PNG strings)
# ---------------------------------------------------------------------------

def _fig_to_b64(fig: plt.Figure) -> str:
    """Encode a matplotlib figure as a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _plot_check_summary(report: FrameworkReport) -> str:
    """Horizontal bar chart: passed vs failed per phase."""
    phases = {
        "Quality": report.quality_results,
        "Leakage": report.leakage_results,
        "Impact": report.impact_results,
    }
    labels, passed_counts, failed_counts = [], [], []
    for phase, results in phases.items():
        if not results:
            continue
        labels.append(phase)
        passed_counts.append(sum(1 for r in results if r.passed))
        failed_counts.append(sum(1 for r in results if not r.passed))

    if not labels:
        return ""

    fig, ax = plt.subplots(figsize=(7, max(2, len(labels) * 0.8 + 1)))
    y = range(len(labels))
    ax.barh(list(y), passed_counts, color="#66bb6a", label="Passed")
    ax.barh(list(y), failed_counts, left=passed_counts, color="#ef5350", label="Failed")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of checks")
    ax.set_title("Check results by phase")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_severity_distribution(report: FrameworkReport) -> str:
    """Pie chart showing severity breakdown of all failed checks."""
    severity_counts: dict[str, int] = {"info": 0, "warning": 0, "error": 0}
    for r in report.all_results():
        severity_counts[r.severity] = severity_counts.get(r.severity, 0) + 1

    non_zero = {k: v for k, v in severity_counts.items() if v > 0}
    if not non_zero:
        return ""

    colours = {"info": "#64b5f6", "warning": "#ffb74d", "error": "#e57373"}
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        list(non_zero.values()),
        labels=[k.upper() for k in non_zero],
        colors=[colours.get(k, "#aaa") for k in non_zero],
        autopct="%1.0f%%",
        startangle=140,
    )
    ax.set_title("Severity distribution")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_impact_comparison(report: FrameworkReport) -> str:
    """Grouped bar chart comparing baseline vs cleaned scores per model."""
    if not report.impact_results:
        return ""

    models, baselines, cleaned_scores = [], [], []
    for r in report.impact_results:
        d = r.details
        if "baseline_score" in d and "cleaned_score" in d:
            models.append(d.get("model", r.check_name))
            baselines.append(d["baseline_score"])
            cleaned_scores.append(d["cleaned_score"])

    if not models:
        return ""

    x = range(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(5, len(models) * 1.5 + 2), 4))
    ax.bar([i - width / 2 for i in x], baselines, width, label="Baseline", color="#42a5f5")
    ax.bar([i + width / 2 for i in x], cleaned_scores, width, label="Cleaned", color="#66bb6a")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=15, ha="right")
    metric = report.impact_results[0].details.get("metric", "score") if report.impact_results else "score"
    ax.set_ylabel(metric)
    ax.set_title("Model performance: baseline vs. cleaned")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    report: FrameworkReport,
    output_dir: str | Path,
    config: dict,
) -> Path:
    """Render *report* to an HTML file inside *output_dir*.

    Args:
        report: Fully (or partially) populated FrameworkReport.
        output_dir: Directory where the HTML file will be written.
        config: The ``reporting`` config block.

    Returns:
        Path to the generated HTML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    include_plots: bool = config.get("include_plots", True)

    plots: dict[str, str] = {}
    if include_plots:
        logger.debug("Generating report plots…")
        _safe_add_plot(plots, "Check results by phase", _plot_check_summary, report)
        _safe_add_plot(plots, "Severity distribution", _plot_severity_distribution, report)
        _safe_add_plot(plots, "Impact analysis — baseline vs. cleaned",
                       _plot_impact_comparison, report)

    env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), autoescape=False)
    template = env.get_template("report.html.j2")
    html = template.render(report=report, plots=plots)

    timestamp = report.run_timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"report_{report.dataset_name}_{timestamp}.html"
    output_path = output_dir / filename
    output_path.write_text(html, encoding="utf-8")

    logger.info(f"HTML report written to '{output_path}' ({output_path.stat().st_size // 1024} KB)")
    return output_path


def _safe_add_plot(
    plots: dict[str, str],
    title: str,
    fn,
    report: FrameworkReport,
) -> None:
    """Call *fn(report)* and store result in *plots*; silently skip on error."""
    try:
        img = fn(report)
        if img:
            plots[title] = img
    except Exception as exc:
        logger.warning(f"Plot '{title}' failed: {exc}")
