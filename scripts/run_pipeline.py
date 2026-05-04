"""End-to-end experiment runner — Phase 7.

Runs the full ml-framework pipeline on each of the three synthetic datasets
(clean, dirty, leaky) and prints a comparison table showing what was detected
and how much model performance changed.

Usage:
    python scripts/run_pipeline.py

The datasets are auto-generated if not present.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import load_config
from src.impact_analysis import run_impact_analysis
from src.leakage_checks import run_all_leakage_checks
from src.quality_checks import run_all_quality_checks
from src.reporting import generate_report
from src.utils import CheckResult, FrameworkReport, load_dataset, setup_logger

DATA_DIR   = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"
CONFIG     = ROOT / "configs" / "config.yaml"


def _ensure_datasets() -> None:
    datasets = ["clean_dataset.csv", "dirty_dataset.csv", "leaky_dataset.csv"]
    if not all((DATA_DIR / d).exists() for d in datasets):
        print("Generating synthetic datasets…")
        from scripts.generate_data import main as gen
        gen()


def _run_single(name: str, csv_path: Path, target_col: str) -> FrameworkReport:
    config = load_config(CONFIG)

    # Point config at the right dataset
    config["dataset"]["path"] = str(csv_path)
    config["dataset"]["target_column"] = target_col

    # Faster settings for the experiment
    config["impact_analysis"]["models"] = ["logistic_regression"]
    config["impact_analysis"]["cv_folds"] = 3

    df = load_dataset(csv_path)

    report = FrameworkReport(
        dataset_name=name,
        metadata={
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
            "target_col": target_col,
        },
    )

    report.quality_results  = run_all_quality_checks(df, target_col, config["quality_checks"])
    report.leakage_results  = run_all_leakage_checks(df, target_col, config["leakage_checks"])
    report.impact_results   = run_impact_analysis(df, target_col, report, config["impact_analysis"])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    generate_report(report, REPORT_DIR, config["reporting"])
    return report


def _print_summary(reports: dict[str, FrameworkReport]) -> None:
    print("\n" + "=" * 72)
    print("  EXPERIMENT RESULTS SUMMARY")
    print("=" * 72)
    header = f"{'Dataset':<22} {'Total':>5} {'Pass':>5} {'Fail':>5} {'Err':>4} {'Warn':>5}"
    print(header)
    print("-" * 72)
    for name, report in reports.items():
        s = report.summary()
        print(
            f"{name:<22} {s['total_checks']:>5} {s['passed']:>5} {s['failed']:>5} "
            f"{s['errors']:>4} {s['warnings']:>5}"
        )
    print("=" * 72)

    print("\n  IMPACT ANALYSIS (logistic regression, accuracy)\n")
    for name, report in reports.items():
        for r in report.impact_results:
            d = r.details
            status = "✓" if r.passed else "✗"
            print(
                f"  {status} {name:<22}  "
                f"baseline={d.get('baseline_score', 0):.3f}  "
                f"cleaned={d.get('cleaned_score', 0):.3f}  "
                f"Δ={d.get('delta', 0):+.3f}  "
                f"dropped={d.get('dropped_columns', [])}"
            )
    print()

    print("  LEAKAGE FAILURES\n")
    for name, report in reports.items():
        failed = [r for r in report.leakage_results if not r.passed]
        if failed:
            for r in failed:
                print(f"  ✗ [{name}] {r.check_name}: {r.message}")
        else:
            print(f"  ✓ [{name}] No leakage detected")
    print()

    print(f"  HTML reports saved to: {REPORT_DIR}/")
    print("=" * 72 + "\n")


def main() -> int:
    setup_logger(log_level="WARNING")   # suppress verbose logging during demo
    _ensure_datasets()

    experiments = {
        "clean_dataset":  (DATA_DIR / "clean_dataset.csv",  "target"),
        "dirty_dataset":  (DATA_DIR / "dirty_dataset.csv",  "target"),
        "leaky_dataset":  (DATA_DIR / "leaky_dataset.csv",  "target"),
    }

    reports: dict[str, FrameworkReport] = {}
    for name, (path, target_col) in experiments.items():
        print(f"\nRunning pipeline on: {name}…")
        reports[name] = _run_single(name, path, target_col)

    _print_summary(reports)
    return 0


if __name__ == "__main__":
    sys.exit(main())
