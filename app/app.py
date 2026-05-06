"""Streamlit web application for interactive dataset readiness analysis (Phase 12).

Run with:
    streamlit run app/app.py
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Make src importable from app/ directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.checker import DatasetChecker

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ML Dataset Readiness Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_COLOURS = {"info": "#64b5f6", "warning": "#ffb74d", "error": "#ef5350"}
PRIORITY_COLOURS = {"high": "#ef5350", "medium": "#ffb74d", "low": "#66bb6a"}
GRADE_COLOURS    = {"A": "#2e7d32", "B": "#558b2f", "C": "#f57f17",
                    "D": "#e65100", "F": "#b71c1c"}


def _load_df(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def _badge(text: str, colour: str) -> str:
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 8px;'
        f'border-radius:10px;font-size:.8rem;font-weight:600;">{text}</span>'
    )


def _results_table(results: list) -> None:
    if not results:
        st.caption("No checks in this phase.")
        return
    rows = []
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        rows.append({
            "Check": r.check_name,
            "Status": status,
            "Severity": r.severity.upper(),
            "Message": r.message,
            "Affected columns": ", ".join(r.affected_columns) if r.affected_columns else "—",
        })
    df_table = pd.DataFrame(rows)
    st.dataframe(
        df_table.style.apply(
            lambda col: [
                "color: #2e7d32" if v == "✓ PASS"
                else "color: #c62828" if v == "✗ FAIL"
                else "" for v in col
            ],
            subset=["Status"],
        ),
        use_container_width=True,
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — upload & config
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 Dataset Readiness")
    st.caption("Upload a dataset and configure the analysis.")

    uploaded_file = st.file_uploader(
        "Dataset file",
        type=["csv", "parquet", "xlsx", "xls"],
        help="CSV, Parquet, or Excel",
    )

    if uploaded_file:
        with st.spinner("Loading…"):
            try:
                df_raw = _load_df(uploaded_file)
                st.success(f"{df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")
            except Exception as exc:
                st.error(f"Could not load file: {exc}")
                df_raw = None
    else:
        df_raw = None
        st.info("Upload a file to continue.")

    if df_raw is not None:
        st.divider()
        st.subheader("Configuration")

        target_col = st.selectbox(
            "Target column ⭐",
            df_raw.columns.tolist(),
            help="The column you want to predict.",
        )

        date_col_opts = ["(none)"] + df_raw.columns.tolist()
        date_col = st.selectbox(
            "Date column (optional)",
            date_col_opts,
            help="For temporal leakage and drift detection.",
        )
        date_col = None if date_col == "(none)" else date_col

        corr_thr = st.slider(
            "Leakage correlation threshold", 0.80, 1.00, 0.95, 0.01,
            help="Features with |r| ≥ this value vs. target are flagged as leaky.",
        )

        run_impact = st.checkbox(
            "Run impact analysis (slow)",
            value=True,
            help="Trains ML models to quantify the effect of issues.",
        )

        models = st.multiselect(
            "Models",
            ["logistic_regression", "random_forest", "xgboost"],
            default=["logistic_regression"],
            disabled=not run_impact,
        )

        st.divider()
        run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────

st.title("ML Dataset Readiness Checker")
st.caption(
    "Automated framework for dataset quality assessment, leakage detection, "
    "and ML readiness scoring."
)

if df_raw is None:
    st.info("👈 Upload a dataset in the sidebar to get started.")
    st.stop()

# Run analysis
if run_btn or "report" not in st.session_state:
    if run_btn:
        with st.spinner("Running analysis — this may take 30–60 s…"):
            checker = DatasetChecker()
            checker.set(
                **{"leakage_checks__target_leakage__correlation_threshold": corr_thr},
                **{"impact_analysis__models": models},
            )
            if date_col:
                checker.set(**{
                    "leakage_checks__temporal_leakage__date_column": date_col,
                    "drift_checks__covariate_drift__date_column": date_col,
                    "drift_checks__label_drift__date_column": date_col,
                })
            report = checker.run(df_raw, target_col=target_col, skip_impact=not run_impact)
            st.session_state["report"]  = report
            st.session_state["checker"] = checker
        st.success("Analysis complete!")

if "report" not in st.session_state:
    st.info("Configure the analysis in the sidebar and click **Run Analysis**.")
    st.stop()

report  = st.session_state["report"]
checker = st.session_state["checker"]
rs      = report.readiness_score

# ── Score banner ─────────────────────────────────────────────────────────────
if rs:
    grade_colour = GRADE_COLOURS.get(rs.grade, "#555")
    col_score, col_q, col_l, col_f, col_s = st.columns(5)
    with col_score:
        st.metric("Overall Score", f"{rs.overall}/100")
        st.markdown(
            f'<p style="font-size:2.5rem;font-weight:800;color:{grade_colour};'
            f'margin:0">{rs.grade}</p><p style="color:#555;font-size:.85rem">'
            f'{rs.label}</p>',
            unsafe_allow_html=True,
        )
    with col_q:
        st.metric("Quality", f"{rs.quality.score}/100",
                  f"{rs.quality.checks_passed}/{rs.quality.checks_total} passed")
    with col_l:
        st.metric("Leakage", f"{rs.leakage.score}/100",
                  f"{rs.leakage.checks_passed}/{rs.leakage.checks_total} passed")
    with col_f:
        st.metric("Features", f"{rs.features.score}/100",
                  f"{rs.features.checks_passed}/{rs.features.checks_total} passed")
    with col_s:
        st.metric("Sufficiency", f"{rs.sufficiency.score}/100",
                  f"{rs.sufficiency.checks_passed}/{rs.sufficiency.checks_total} passed")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_quality, tab_leakage, tab_features, tab_sufficiency, tab_drift, tab_recs, tab_dl = st.tabs([
    "🔍 Quality",
    "🔒 Leakage",
    "📊 Features",
    "📏 Sufficiency",
    "📉 Drift",
    "💡 Recommendations",
    "📥 Download",
])

with tab_quality:
    st.subheader("Quality Checks")
    _results_table(report.quality_results)

with tab_leakage:
    st.subheader("Leakage Checks")
    _results_table(report.leakage_results)

with tab_features:
    st.subheader("Feature Analysis")
    _results_table(report.feature_results)
    if report.impact_results:
        st.subheader("Impact Analysis")
        _results_table(report.impact_results)

with tab_sufficiency:
    st.subheader("Statistical Sufficiency")
    _results_table(report.sufficiency_results)

with tab_drift:
    st.subheader("Distribution Drift")
    _results_table(report.drift_results)

with tab_recs:
    st.subheader("Recommendations")
    if not report.recommendations:
        st.success("No recommendations — dataset looks clean!")
    else:
        priority_filter = st.radio(
            "Filter by priority",
            ["All", "High", "Medium", "Low"],
            horizontal=True,
        )
        for rec in report.recommendations:
            if priority_filter != "All" and rec.priority.lower() != priority_filter.lower():
                continue
            colour = PRIORITY_COLOURS.get(rec.priority, "#aaa")
            with st.expander(f"[{rec.priority.upper()}] {rec.check_name} — {rec.action}"):
                st.markdown(f"**Why:** {rec.rationale}")
                if rec.code_snippet:
                    st.code(rec.code_snippet, language="python")

with tab_dl:
    st.subheader("Download Report")
    with tempfile.TemporaryDirectory() as tmp:
        report_path = checker.save_report(tmp)
        html_bytes  = report_path.read_bytes()

    st.download_button(
        label="📥  Download HTML Report",
        data=html_bytes,
        file_name=report_path.name,
        mime="text/html",
        use_container_width=True,
    )
    st.caption(f"Report: {report_path.name} ({len(html_bytes) // 1024} KB)")

    st.subheader("Export as JSON")
    import json
    result_dict = checker.to_dict()
    st.download_button(
        label="📥  Download JSON Results",
        data=json.dumps(result_dict, indent=2, default=str),
        file_name=f"results_{report.dataset_name}.json",
        mime="application/json",
        use_container_width=True,
    )
