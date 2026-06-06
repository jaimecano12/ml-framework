"""Real-world leakage and quality case studies.

Runs the ml-framework pipeline on three real-world datasets and extracts
the most informative findings for each, saving a structured JSON report.

Usage
-----
    python scripts/case_studies.py

Output
------
    reports/case_studies.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.checker import DatasetChecker

DATASETS = [
    {
        "name":       "Titanic",
        "path":       "data/raw/titanic.csv",
        "target_col": "survived",
        "config":     "configs/titanic_config.yaml",
    },
    {
        "name":       "Adult Census Income",
        "path":       "data/raw/adult.csv",
        "target_col": "income",
        "config":     "configs/config.yaml",
    },
    {
        "name":       "German Credit",
        "path":       "data/raw/german_credit.csv",
        "target_col": "credit_risk",
        "config":     "configs/config.yaml",
    },
]

OUTPUT_PATH = Path("reports/case_studies.json")


def _top_findings(checker: DatasetChecker, n: int = 5) -> list[dict]:
    """Return the n most actionable failed checks."""
    failed = checker.failed_checks()
    severity_order = {"error": 0, "warning": 1, "info": 2}
    failed.sort(key=lambda r: severity_order.get(r.severity, 3))
    results = []
    for r in failed[:n]:
        entry = {
            "check": r.check_name,
            "severity": r.severity,
            "message": r.message,
        }
        if r.affected_columns:
            entry["affected_columns"] = r.affected_columns[:5]
        results.append(entry)
    return results


def run_case_study(dataset: dict) -> dict:
    path = Path(dataset["path"])
    if not path.exists():
        return {"name": dataset["name"], "error": f"Dataset not found: {path}"}

    print(f"\n[{dataset['name']}] Running pipeline...")
    checker = DatasetChecker(dataset["config"])
    checker.run(dataset["path"], target_col=dataset["target_col"], skip_impact=True)

    summary = checker.summary()
    report  = checker._report
    top_recs = checker.top_recommendations(priority="high")

    return {
        "name":            dataset["name"],
        "rows":            report.metadata.get("shape", [0])[0],
        "cols":            report.metadata.get("shape", [0, 0])[1],
        "target_col":      dataset["target_col"],
        "readiness_score": checker.score,
        "grade":           checker.grade,
        "checks_passed":   summary.get("passed", 0),
        "checks_total":    summary.get("total_checks", 0),
        "top_findings":    _top_findings(checker),
        "top_recommendations": [
            {"action": r.action, "rationale": r.rationale}
            for r in top_recs[:3]
        ],
    }


def main() -> None:
    results = []
    for ds in DATASETS:
        result = run_case_study(ds)
        results.append(result)
        score = result.get("readiness_score", "N/A")
        grade = result.get("grade", "N/A")
        findings = len(result.get("top_findings", []))
        print(f"  Score={score}/100  Grade={grade}  Top findings={findings}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nCase studies saved to {OUTPUT_PATH}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"  {'Dataset':<24} {'Rows':>7} {'Score':>6} {'Grade':>6} {'Pass':>8}")
    print("  " + "-" * 54)
    for r in results:
        if "error" not in r:
            passed = r.get("checks_passed", 0)
            total  = r.get("checks_total", 0)
            print(
                f"  {r['name']:<24} {r['rows']:>7,} {r['readiness_score']:>6.1f} "
                f"{r['grade']:>6}  {passed}/{total}"
            )
    print("=" * 70)


if __name__ == "__main__":
    main()
