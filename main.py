"""Entry point for the ML Framework CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    logger.info(f"Config   : {args.config}")
    logger.info(f"Dataset  : {args.dataset}")
    logger.info(f"Output   : {args.output_dir}")

    # Phase 2+: load config
    # Phase 3+: quality checks
    # Phase 4+: leakage checks
    # Phase 5+: impact analysis
    # Phase 6+: report generation

    logger.info("Run complete (no checks configured yet — see Phase 2+)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
