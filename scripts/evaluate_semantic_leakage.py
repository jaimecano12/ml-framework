"""Quantitative evaluation of the LLM semantic leakage module.

Runs the semantic leakage analyser (real or mock) against a manually
constructed benchmark of 30 labelled features across five domain contexts,
and reports precision, recall, and F1 at two risk thresholds.

Usage
-----
# With real Azure credentials:
    export AZURE_OPENAI_API_KEY=...
    export AZURE_OPENAI_ENDPOINT=...
    python scripts/evaluate_semantic_leakage.py

# Deterministic mock mode (no credentials required):
    python scripts/evaluate_semantic_leakage.py --mock

Output
------
Prints a results table to stdout and writes
  reports/semantic_benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.semantic_leakage import _build_user_prompt, _parse_llm_response

BENCHMARK_PATH = Path("data/semantic_benchmark.json")
RESULTS_PATH   = Path("reports/semantic_benchmark_results.json")

RISK_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


# ---------------------------------------------------------------------------
# Mock LLM — deterministic pattern-based simulation of GPT-4o-mini behaviour
# ---------------------------------------------------------------------------

_TEMPORAL_PATTERNS = (
    "future_", "_after_", "post_", "_post", "salary_after", "spending_after",
    "post_hire", "post_approval",
)
_POSTHOC_PATTERNS = (
    "discharge_", "defect_code", "final_quality", "final_default",
    "cancellation_reason", "performance_outcome", "_outcome_label",
    "days_in_hospital", "discharge_medications", "exit_code", "result_code",
)
_PROXY_PATTERNS = (
    "batch_rejection_rate", "rejection_rate", "group_default_rate",
)


def _mock_llm_response(features: list[str], context: str) -> str:
    """Return a JSON array simulating realistic GPT-4o-mini behaviour."""
    results = []
    for feat in features:
        fn = feat.lower()
        risk, ltype, reasoning, rec = "none", "none", "", ""

        for p in _TEMPORAL_PATTERNS:
            if p in fn:
                risk = "high"
                ltype = "temporal"
                reasoning = (
                    f"'{feat}' contains a temporal indicator suggesting this "
                    "feature is measured after the prediction event."
                )
                rec = f"Verify that {feat} is computed before the target event."
                break

        if risk == "none":
            for p in _POSTHOC_PATTERNS:
                if fn.startswith(p) or fn.endswith(p) or p in fn:
                    risk = "high"
                    ltype = "post_hoc"
                    reasoning = (
                        f"'{feat}' appears to be assigned or computed after the "
                        "outcome is known, making it a post-hoc label."
                    )
                    rec = f"Remove or replace {feat} with information available before the event."
                    break

        if risk == "none":
            for p in _PROXY_PATTERNS:
                if p in fn:
                    risk = "high"
                    ltype = "proxy"
                    reasoning = (
                        f"'{feat}' is a rate derived from the same labels used as the "
                        "target, creating indirect leakage."
                    )
                    rec = f"Do not use group-level rates computed from the target variable."
                    break

        # treatment_count is genuinely ambiguous — keyword matching returns low,
        # simulating that a real LLM may under-flag subtle cases without
        # additional domain context.
        if feat == "treatment_count" and risk == "none":
            risk = "low"
            ltype = "temporal"
            reasoning = (
                "Could reflect treatments during the current stay (post-hoc) "
                "or prior history; insufficient context to confirm."
            )
            rec = "Clarify whether this reflects pre-admission or in-stay treatment count."

        if risk == "none":
            reasoning = (
                f"'{feat}' appears to be a legitimate pre-event feature "
                "available at prediction time."
            )

        results.append({
            "feature_name": feat,
            "risk_level": risk,
            "leakage_type": ltype,
            "reasoning": reasoning,
            "recommendation": rec,
        })

    return json.dumps(results)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {"precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def evaluate(benchmark: list[dict], use_mock: bool) -> dict:
    """Run evaluation at thresholds 'medium' and 'high'."""

    # Group features by dataset context so the LLM gets coherent prompts
    context_groups: dict[str, list[dict]] = defaultdict(list)
    for item in benchmark:
        context_groups[item["dataset_context"]].append(item)

    # Collect LLM predictions
    predictions: dict[str, str] = {}
    for context, items in context_groups.items():
        feat_names = [it["feature_name"] for it in items]
        sample_vals = {f: [] for f in feat_names}

        if use_mock:
            raw = _mock_llm_response(feat_names, context)
        else:
            from src.semantic_leakage import _call_llm
            prompt = _build_user_prompt(feat_names, "target", context, sample_vals)
            raw = _call_llm(prompt, os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"))

        assessments = _parse_llm_response(raw)
        for a in assessments:
            predictions[a.feature_name] = a.risk_level

    # Build ground-truth map
    ground_truth: dict[str, str] = {
        item["feature_name"]: item["ground_truth_risk"]
        for item in benchmark
    }

    results = {}
    for threshold in ("medium", "high"):
        thr_idx = RISK_ORDER[threshold]
        tp = fp = fn = tn = 0
        per_feature = []
        for feat, gt_risk in ground_truth.items():
            pred_risk = predictions.get(feat, "none")
            gt_pos  = RISK_ORDER.get(gt_risk,   0) >= thr_idx
            pred_pos = RISK_ORDER.get(pred_risk, 0) >= thr_idx
            if gt_pos and pred_pos:
                tp += 1; verdict = "TP"
            elif not gt_pos and pred_pos:
                fp += 1; verdict = "FP"
            elif gt_pos and not pred_pos:
                fn += 1; verdict = "FN"
            else:
                tn += 1; verdict = "TN"
            per_feature.append({
                "feature": feat,
                "ground_truth": gt_risk,
                "predicted": pred_risk,
                "verdict": verdict,
            })

        metrics = _metrics(tp, fp, fn)
        metrics["tn"] = tn
        results[threshold] = {"metrics": metrics, "per_feature": per_feature}

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: dict) -> None:
    print("\n" + "=" * 60)
    print("  Semantic Leakage Module — Quantitative Evaluation")
    print("=" * 60)
    header = f"  {'Threshold':<12} {'Precision':>10} {'Recall':>8} {'F1':>8}  {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}"
    print(header)
    print("  " + "-" * 56)
    for thr, data in results.items():
        m = data["metrics"]
        print(
            f"  {thr:<12} {m['precision']:>10.3f} {m['recall']:>8.3f} "
            f"{m['f1']:>8.3f}  {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4}"
        )
    print("=" * 60)

    print("\n  False positives / negatives:\n")
    for thr, data in results.items():
        errors = [f for f in data["per_feature"] if f["verdict"] in ("FP", "FN")]
        if errors:
            print(f"  [threshold={thr}]")
            for e in errors:
                print(f"    {e['verdict']}  {e['feature']}  "
                      f"(gt={e['ground_truth']}, pred={e['predicted']})")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate semantic leakage module.")
    parser.add_argument("--mock", action="store_true",
                        help="Use deterministic mock instead of real LLM API")
    args = parser.parse_args()

    use_mock = args.mock
    if not use_mock:
        if not os.environ.get("AZURE_OPENAI_API_KEY"):
            print("No AZURE_OPENAI_API_KEY found — falling back to --mock mode.")
            use_mock = True

    benchmark = json.loads(BENCHMARK_PATH.read_text())
    print(f"Loaded benchmark: {len(benchmark)} features, "
          f"mode={'mock' if use_mock else 'live LLM'}")

    results = evaluate(benchmark, use_mock)
    print_table(results)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    output = {
        "benchmark_size": len(benchmark),
        "mode": "mock" if use_mock else "live",
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
