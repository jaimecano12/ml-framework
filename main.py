"""Entry point for the ML Framework CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import get_section, load_config
from src.impact_analysis import run_impact_analysis
from src.leakage_checks import run_all_leakage_checks
from src.quality_checks import run_all_quality_checks
from src.reporting import generate_report
from src.utils import FrameworkReport, load_dataset, setup_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ml-framework",
        description="Automated framework for dataset quality assessment and data leakage detection.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to the YAML configuration file (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the input dataset (CSV, Parquet, or Excel).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory where the generated report will be saved (default: reports/).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logger(log_level=args.log_level)

    from loguru import logger

    logger.info("ML Framework starting up")

    config = load_config(args.config)

    # CLI flags override config values when explicitly provided
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

    # Phase 5 — impact analysis
    report.impact_results = run_impact_analysis(
        df, target_col, report, get_section(config, "impact_analysis")
    )

    # Phase 6 — report generation
    output_dir = Path(config["reporting"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = generate_report(report, output_dir, get_section(config, "reporting"))
    logger.info(f"Report saved to: {report_path}")

    summary = report.summary()
    logger.info(
        f"Run complete — {summary['passed']}/{summary['total_checks']} checks passed, "
        f"{summary['errors']} error(s), {summary['warnings']} warning(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
