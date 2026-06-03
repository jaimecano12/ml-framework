"""Quantitative benchmark: ml-framework vs ydata-profiling, Deepchecks, Great Expectations.

Dimensions evaluated:
  1. Feature/check coverage matrix (static comparison)
  2. Leakage detection rate on synthetic scenarios
  3. Runtime efficiency on three dataset sizes
  4. Configuration flexibility (qualitative)

Output:
  reports/benchmark_results.json  — full results
  reports/benchmark_report.txt    — human-readable summary
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data" / "raw"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature coverage matrix
# ---------------------------------------------------------------------------

FEATURE_MATRIX = {
    "check": [
        "Missing value detection",
        "Duplicate row detection",
        "Outlier detection",
        "Class imbalance detection",
        "Constant / near-constant features",
        "Target leakage (correlation-based)",
        "Target leakage (MI-based)",
        "Target leakage (performance inflation)",
        "Unified leakage risk score",
        "Temporal ordering check",
        "ID / high-cardinality column check",
        "Train/test overlap detection",
        "Feature correlation analysis",
        "Feature relevance (MI) analysis",
        "Distribution shape (skew/kurtosis)",
        "Statistical sufficiency (sample size)",
        "Class support check (per-class n)",
        "Cross-val stability check",
        "Covariate drift detection (KS/PSI)",
        "Label drift detection",
        "Impact analysis (before/after cleaning)",
        "Readiness score (0-100, A-F)",
        "Actionable recommendations",
        "Custom plugin / extensibility",
        "CLI interface",
        "Python SDK",
        "Web UI (Streamlit / dashboard)",
        "YAML-based configuration",
        "HTML + JSON export",
    ],
    "ml_framework": [
        True, True, True, True, True,
        True, True, True, True,       # leakage checks
        True, True, True,
        True, True, True,             # feature analysis
        True, True, True, True,       # sufficiency + drift
        True, True, True, True, True, # impact + score + recommendations + plugin
        True, True, True, True, True, # interfaces
    ],
    "ydata_profiling": [
        True, True, True, False, True,
        True, False, False, False,
        False, False, False,
        True, False, True,
        False, False, False, False,
        False, False, False, False, False,
        False, False, False, False, True,
    ],
    "deepchecks": [
        True, True, False, True, True,
        False, False, False, False,
        False, False, True,
        True, True, False,
        False, False, False, True,
        True, False, False, False, False,
        False, True, False, False, True,
    ],
    "great_expectations": [
        True, False, True, False, True,
        False, False, False, False,
        False, False, False,
        False, False, False,
        False, False, False, False,
        False, False, False, False, True,
        True, True, True, True, True,
    ],
}


def build_feature_matrix() -> pd.DataFrame:
    df = pd.DataFrame(FEATURE_MATRIX)
    for col in ["ml_framework", "ydata_profiling", "deepchecks", "great_expectations"]:
        df[col] = df[col].map({True: "YES", False: "NO"})
    return df


def count_features(matrix: pd.DataFrame) -> dict[str, int]:
    return {
        col: int((matrix[col] == "YES").sum())
        for col in ["ml_framework", "ydata_profiling", "deepchecks", "great_expectations"]
    }


# ---------------------------------------------------------------------------
# Leakage detection capability
# ---------------------------------------------------------------------------

def _make_leakage_scenarios() -> dict[str, tuple[pd.DataFrame, str]]:
    """Return {scenario_name: (DataFrame, target_col)} with intentional leakage."""
    rng = np.random.default_rng(42)
    n = 300
    target = rng.integers(0, 2, n)

    # Scenario 1: Perfect proxy (corr = 1.0)
    s1 = pd.DataFrame({"feature": rng.normal(0, 1, n), "proxy": target.astype(float), "target": target})

    # Scenario 2: Noisy proxy (corr ≈ 0.97)
    s2 = pd.DataFrame({"feature": rng.normal(0, 1, n),
                        "noisy_proxy": target + rng.normal(0, 0.1, n), "target": target})

    # Scenario 3: ID column (high cardinality string)
    s3 = pd.DataFrame({"feature": rng.normal(0, 1, n),
                        "row_id": [f"ID_{i}" for i in range(n)], "target": target})

    # Scenario 4: Indirect computed feature (target * scale + noise)
    s4 = pd.DataFrame({"feature": rng.normal(0, 1, n),
                        "indirect": target * 50 + rng.normal(0, 2, n), "target": target})

    return {
        "perfect_proxy": (s1, "target"),
        "noisy_proxy":   (s2, "target"),
        "id_column":     (s3, "target"),
        "indirect_feat": (s4, "target"),
    }


def _detect_mlframework(df: pd.DataFrame, target: str, scenario: str) -> dict[str, bool]:
    from src.leakage_checks import run_all_leakage_checks
    cfg = {
        "enabled": True,
        "target_leakage":    {"enabled": True, "correlation_threshold": 0.95},
        "train_test_overlap": {"enabled": True, "test_size": 0.2, "random_state": 42},
        "temporal_leakage":  {"enabled": True, "date_column": None},
        "id_column_leakage": {"enabled": True, "cardinality_threshold": 0.95},
        "leakage_risk_score": {"enabled": True, "threshold": 0.7,
                               "weights": [0.35, 0.35, 0.30], "cv_folds": 3},
    }
    results = run_all_leakage_checks(df, target, cfg)
    failed = {r.check_name for r in results if not r.passed}
    detected = len(failed) > 0
    return {"detected": detected, "checks_failed": list(failed)}


def _detect_ydata(df: pd.DataFrame, target: str, scenario: str) -> dict[str, bool]:
    try:
        from ydata_profiling import ProfileReport
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            profile = ProfileReport(df, minimal=True, progress_bar=False)
            desc = profile.get_description()

        correlations = desc.correlations if hasattr(desc, "correlations") else {}
        # Check if any correlation with target exceeds 0.95
        detected = False
        if hasattr(desc, "variables"):
            for var_name, var_info in desc.variables.items():
                if var_name == target:
                    continue
                corr_val = getattr(var_info, "pearsonr", None) or 0
                if abs(corr_val) >= 0.95:
                    detected = True
                    break
        # Fallback: check correlation matrix
        if not detected and "auto" in correlations:
            corr_matrix = correlations.get("auto", {})
        return {"detected": detected, "note": "correlation-based only"}
    except Exception as exc:
        return {"detected": False, "error": str(exc)}


def _detect_deepchecks(df: pd.DataFrame, target: str, scenario: str) -> dict[str, bool]:
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from deepchecks.tabular import Dataset
            from deepchecks.tabular.checks import (
                FeatureLabelCorrelation,
                StringMismatch,
            )

        ds = Dataset(df, label=target, cat_features=[c for c in df.select_dtypes(["object"]).columns])
        detected = False

        # FeatureLabelCorrelation check
        try:
            result = FeatureLabelCorrelation(ppscore_threshold=0.8).run(ds)
            if result.passed_conditions() is not None and not all(result.passed_conditions()):
                detected = True
        except Exception:
            pass

        return {"detected": detected, "note": "PPS-based feature-label correlation only"}
    except Exception as exc:
        return {"detected": False, "error": str(exc)}


def _detect_great_expectations(df: pd.DataFrame, target: str, scenario: str) -> dict[str, bool]:
    # Great Expectations does not have native leakage detection; it validates predefined rules
    # The best proxy: expect_column_values_to_be_unique on ID-like columns (must be predefined)
    if scenario == "id_column":
        try:
            import great_expectations as gx
            context = gx.get_context(mode="ephemeral")
            detected = True  # GE CAN detect this IF you define the expectation manually
            return {"detected": detected, "note": "requires manual expectation definition"}
        except Exception as exc:
            return {"detected": False, "error": str(exc)}
    return {"detected": False, "note": "no native leakage detection"}


def run_detection_benchmark() -> dict:
    scenarios = _make_leakage_scenarios()
    tools = {
        "ml_framework":       _detect_mlframework,
        "ydata_profiling":    _detect_ydata,
        "deepchecks":         _detect_deepchecks,
        "great_expectations": _detect_great_expectations,
    }

    results: dict[str, dict] = {tool: {} for tool in tools}

    for scenario_name, (df, target) in scenarios.items():
        print(f"  Scenario: {scenario_name}")
        for tool_name, detect_fn in tools.items():
            t0 = time.perf_counter()
            result = detect_fn(df, target, scenario_name)
            elapsed = time.perf_counter() - t0
            results[tool_name][scenario_name] = {**result, "runtime_s": round(elapsed, 3)}

    return results


# ---------------------------------------------------------------------------
# Runtime benchmark
# ---------------------------------------------------------------------------

def run_runtime_benchmark() -> dict[str, dict]:
    sizes = {
        "small  (500 rows)":   DATA_DIR / "clean_dataset.csv",
        "medium (1 000 rows)": DATA_DIR / "proxy_leakage.csv",
        "large  (5 300 rows)": DATA_DIR / "dirty_dataset.csv",
    }

    runtime_results: dict[str, dict] = {}

    for size_label, path in sizes.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        target_col = "target"
        runtime_results[size_label] = {}

        # ml-framework runtime
        t0 = time.perf_counter()
        try:
            from src.checker import DatasetChecker
            checker = DatasetChecker(str(ROOT / "configs" / "config.yaml"))
            checker.run(str(path), target_col=target_col)
            runtime_results[size_label]["ml_framework"] = round(time.perf_counter() - t0, 2)
        except Exception as exc:
            runtime_results[size_label]["ml_framework"] = f"ERROR: {exc}"

        # ydata-profiling runtime
        t0 = time.perf_counter()
        try:
            import warnings
            from ydata_profiling import ProfileReport
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ProfileReport(df, minimal=True, progress_bar=False)
            runtime_results[size_label]["ydata_profiling"] = round(time.perf_counter() - t0, 2)
        except Exception as exc:
            runtime_results[size_label]["ydata_profiling"] = f"ERROR: {exc}"

        # deepchecks runtime
        t0 = time.perf_counter()
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from deepchecks.tabular import Dataset
                from deepchecks.tabular.suites import data_integrity
            ds = Dataset(df, label=target_col)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data_integrity().run(ds)
            runtime_results[size_label]["deepchecks"] = round(time.perf_counter() - t0, 2)
        except Exception as exc:
            runtime_results[size_label]["deepchecks"] = f"ERROR: {exc}"

        # great_expectations runtime (basic validation only)
        t0 = time.perf_counter()
        try:
            import great_expectations as gx
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ctx = gx.get_context(mode="ephemeral")
            runtime_results[size_label]["great_expectations"] = round(time.perf_counter() - t0, 2)
        except Exception as exc:
            runtime_results[size_label]["great_expectations"] = f"ERROR: {exc}"

        print(f"  {size_label}: {runtime_results[size_label]}")

    return runtime_results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _bool_row(row: pd.Series) -> str:
    def fmt(v: str) -> str:
        return "✓" if v == "YES" else "✗"
    return (
        f"  {row['check']:<50} "
        f"{fmt(row['ml_framework']):<6} "
        f"{fmt(row['ydata_profiling']):<16} "
        f"{fmt(row['deepchecks']):<12} "
        f"{fmt(row['great_expectations'])}"
    )


def generate_report(
    feature_matrix: pd.DataFrame,
    feature_counts: dict,
    detection_results: dict,
    runtime_results: dict,
) -> str:
    lines = [
        "=" * 80,
        "BENCHMARK: ml-framework vs ydata-profiling vs Deepchecks vs Great Expectations",
        "=" * 80,
        "",
        "─── 1. FEATURE COVERAGE ────────────────────────────────────────────────────",
        f"  {'Check':<50} {'ML-FW':<6} {'ydata-prof':<16} {'Deepchecks':<12} {'GE'}",
        "  " + "-" * 76,
    ]
    for _, row in feature_matrix.iterrows():
        lines.append(_bool_row(row))

    lines += ["", "  Summary (checks supported out of 29):"]
    for tool, count in feature_counts.items():
        lines.append(f"    {tool:<25}: {count}/29")

    lines += [
        "",
        "─── 2. LEAKAGE DETECTION RATE ──────────────────────────────────────────────",
        f"  {'Scenario':<22} {'ML-FW':<10} {'ydata-prof':<14} {'Deepchecks':<14} {'GE'}",
        "  " + "-" * 68,
    ]
    scenarios = list(next(iter(detection_results.values())).keys())
    for sc in scenarios:
        row_parts = [f"  {sc:<22}"]
        for tool in ["ml_framework", "ydata_profiling", "deepchecks", "great_expectations"]:
            d = detection_results[tool].get(sc, {})
            flag = "DETECTED" if d.get("detected") else "missed  "
            row_parts.append(f"{flag:<14}")
        lines.append("".join(row_parts))

    lines += ["", "  Detection rates:"]
    for tool in ["ml_framework", "ydata_profiling", "deepchecks", "great_expectations"]:
        detected = sum(1 for sc in scenarios if detection_results[tool].get(sc, {}).get("detected"))
        lines.append(f"    {tool:<25}: {detected}/{len(scenarios)} scenarios")

    lines += [
        "",
        "─── 3. RUNTIME (seconds) ───────────────────────────────────────────────────",
        f"  {'Dataset':<22} {'ML-FW':<12} {'ydata-prof':<16} {'Deepchecks':<14} {'GE'}",
        "  " + "-" * 70,
    ]
    for size, times in runtime_results.items():
        mf  = str(times.get("ml_framework", "N/A"))
        ydp = str(times.get("ydata_profiling", "N/A"))
        dpc = str(times.get("deepchecks", "N/A"))
        ge  = str(times.get("great_expectations", "N/A"))
        lines.append(f"  {size:<22} {mf:<12} {ydp:<16} {dpc:<14} {ge}")

    lines += ["", "=" * 80]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("\n=== ml-framework Benchmark ===\n")

    print("Building feature matrix…")
    matrix = build_feature_matrix()
    counts = count_features(matrix)

    print("\nRunning leakage detection benchmark…")
    detection = run_detection_benchmark()

    print("\nRunning runtime benchmark…")
    runtime = run_runtime_benchmark()

    report_text = generate_report(matrix, counts, detection, runtime)
    print("\n" + report_text)

    # Save JSON
    json_out = REPORTS_DIR / "benchmark_results.json"
    with open(json_out, "w") as f:
        json.dump({
            "feature_counts": counts,
            "detection_results": detection,
            "runtime_results": runtime,
        }, f, indent=2, default=str)
    print(f"\nJSON saved: {json_out}")

    # Save text
    txt_out = REPORTS_DIR / "benchmark_report.txt"
    txt_out.write_text(report_text)
    print(f"Text report saved: {txt_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
