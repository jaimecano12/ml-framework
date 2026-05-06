"""Entry point for the ML Framework CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import get_section, load_config
from src.drift_checks import run_all_drift_checks
from src.feature_analysis import run_all_feature_checks
from src.impact_analysis import run_impact_analysis
from src.leakage_checks import run_all_leakage_checks
from src.plugins import load_plugins
from src.quality_checks import run_all_quality_checks
from src.recommendations import generate_recommendations
from src.reporting import generate_report
from src.scoring import compute_readiness_score
from src.sufficiency import run_all_sufficiency_checks
from src.utils import FrameworkReport, load_dataset, setup_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ml-framework",
        description="Automated framework for dataset quality assessment and data leakage detection.",
    )
    parser.add_argument("--config",    type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--dataset",   type=Path, required=True)
    parser.add_argument("--output-dir",type=Path, default=Path("reports"))
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logger(log_level=args.log_level)

    from loguru import logger

    logger.info("ML Framework starting up")

    config = load_config(args.config)

    if args.log_level != "INFO":
        config["logging"]["level"] = args.log_level
    if str(args.dataset) != "":
        config["dataset"]["path"] = str(args.dataset)
    if str(args.output_dir) != "reports":
        config["reporting"]["output_dir"] = str(args.output_dir)

    setup_logger(
        log_level=config["logging"]["level"],
        log_file=config["logging"].get("file"),
    )

    logger.info(f"Config      : {args.config}")
    logger.info(f"Dataset     : {config['dataset']['path']}")
    logger.info(f"Target col  : {config['dataset']['target_column']}")
    logger.info(f"Output dir  : {config['reporting']['output_dir']}")

    df = load_dataset(config["dataset"]["path"])
    target_col: str = config["dataset"]["target_column"]

    report = FrameworkReport(
        dataset_name=Path(config["dataset"]["path"]).stem,
        metadata={
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
            "target_col": target_col,
        },
    )

    # Phase 3 — quality checks
    report.quality_results = run_all_quality_checks(
        df, target_col, get_section(config, "quality_checks")
    )

    # Phase 4 — leakage checks
    report.leakage_results = run_all_leakage_checks(
        df, target_col, get_section(config, "leakage_checks")
    )

    # Phase 9 — feature analysis
    report.feature_results = run_all_feature_checks(
        df, target_col, get_section(config, "feature_analysis")
    )

    # Phase 11 — sufficiency checks
    report.sufficiency_results = run_all_sufficiency_checks(
        df, target_col, report, get_section(config, "sufficiency_checks")
    )

    # Phase 14 — drift checks
    report.drift_results = run_all_drift_checks(
        df, target_col, get_section(config, "drift_checks")
    )

    # Phase 15 — plugin checks (appended to each phase's results)
    plugin_paths = config.get("plugins", [])
    if plugin_paths:
        load_plugins(plugin_paths)

    # Phase 5 — impact analysis
    report.impact_results = run_impact_analysis(
        df, target_col, report, get_section(config, "impact_analysis")
    )

    # Phase 8 — recommendations
    report.recommendations = generate_recommendations(report)

    # Phase 10 — readiness score
    report.readiness_score = compute_readiness_score(report)

    # Phase 6 — report generation
    output_dir = Path(config["reporting"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = generate_report(report, output_dir, get_section(config, "reporting"))
    logger.info(f"Report saved to: {report_path}")

    summary = report.summary()
    logger.info(
        f"Run complete — {summary['passed']}/{summary['total_checks']} checks passed "
        f"| Score: {summary.get('readiness_score', 'n/a')}/100 "
        f"Grade: {summary.get('readiness_grade', '?')} "
        f"| {summary['errors']} error(s), {summary['warnings']} warning(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
