# CLAUDE.md — ml-framework (TFM)

## Proyecto

**"An Automated Framework for Dataset Quality Assessment and Data Leakage Detection in Machine Learning"**
TFM de Jaime Cano Moraño — Python 3.13, entorno miniconda (`/opt/miniconda3`).
Repositorio: https://github.com/jaimecano12/ml-framework

---

## Reglas de trabajo

- **Siempre** hacer `git commit + git push` al terminar cada fase o bloque de trabajo.
- Usar `python -m pip install` y `python -m pytest` (nunca `pip` ni `pytest` a secas: apuntan a Python 3.12, no al entorno activo).
- Para ejecutar el notebook usar el kernel `ml-framework`: `python -m jupyter nbconvert --ExecutePreprocessor.kernel_name=ml-framework`
- Un test roto se arregla antes de seguir; nunca se omite con `--ignore` o `xfail` sin justificación.

---

## Estado actual — COMPLETO (15 fases + notebook)

| Fase | Módulo | Tests | Estado |
|------|--------|-------|--------|
| 1 — Scaffold | `src/utils.py`, `main.py` | 16 | ✅ |
| 2 — Config YAML | `src/config.py`, `configs/config.yaml` | 26 | ✅ |
| 3 — Quality checks | `src/quality_checks.py` | 44 | ✅ |
| 4 — Leakage detection | `src/leakage_checks.py` | 41 | ✅ |
| 5 — Impact analysis | `src/impact_analysis.py` | 21 | ✅ |
| 6 — Report generation | `src/reporting.py` + `src/templates/` | 14 | ✅ |
| 7 — Experiments | `scripts/generate_data.py`, `scripts/run_pipeline.py` | — | ✅ |
| 8 — Recommendations | `src/recommendations.py` | 13 | ✅ |
| 9 — Feature analysis | `src/feature_analysis.py` | 22 | ✅ |
| 10 — Readiness score | `src/scoring.py` | 15 | ✅ |
| 11 — Sufficiency | `src/sufficiency.py` | 21 | ✅ |
| 12 — Streamlit app | `app/app.py` | — | ✅ |
| 13 — Python SDK | `src/checker.py` | 14 | ✅ |
| 14 — Drift detection | `src/drift_checks.py` | 14 | ✅ |
| 15 — Plugin system | `src/plugins.py` | 11 | ✅ |
| Demo notebook | `notebooks/framework_demo.ipynb` | — | ✅ |

**Total: 301 tests, 301 passed.**

---

## Arquitectura completa

```
ml-framework/
├── app/
│   └── app.py                   — Streamlit web app (Phase 12)
├── configs/
│   └── config.yaml              — Master config (all phases)
├── data/raw/                    — 5 datasets (3 synthetic + Titanic + Diabetes)
├── notebooks/
│   └── framework_demo.ipynb     — 14-cell executed notebook
├── reports/                     — HTML reports + PNG figures + JSON exports
├── scripts/
│   ├── generate_data.py         — synthetic dataset generator
│   ├── download_real_datasets.py— Titanic + Diabetes via OpenML
│   ├── run_pipeline.py          — end-to-end demo (3 synthetic datasets)
│   ├── build_notebook.py        — notebook builder
│   ├── write_section2.py        — writes Section 2 into tfm.docx
│   └── full_evaluation.py       — comprehensive benchmark
├── src/
│   ├── __init__.py              — public API: DatasetChecker + dataclasses
│   ├── checker.py               — DatasetChecker SDK (Phase 13)
│   ├── config.py                — YAML loader, deep-merge, validation
│   ├── drift_checks.py          — KS + PSI covariate/label drift (Phase 14)
│   ├── feature_analysis.py      — correlation, MI relevance, distribution (Phase 9)
│   ├── impact_analysis.py       — baseline vs cleaned CV comparison (Phase 5)
│   ├── leakage_checks.py        — 4 leakage checks (Phase 4)
│   ├── plugins.py               — @register_check plugin system (Phase 15)
│   ├── quality_checks.py        — 6 quality checks (Phase 3)
│   ├── recommendations.py       — 20 handlers → Recommendation objects (Phase 8)
│   ├── reporting.py             — HTML generation via Jinja2 + matplotlib (Phase 6)
│   ├── scoring.py               — 0-100 ReadinessScore, A-F grade (Phase 10)
│   ├── sufficiency.py           — 4 statistical sufficiency checks (Phase 11)
│   ├── templates/report.html.j2 — HTML template (inline CSS, no deps)
│   └── utils.py                 — CheckResult, FrameworkReport, Recommendation,
│                                   DimensionScore, ReadinessScore, load_dataset
├── tests/                       — 301 tests across 10 test files
├── main.py                      — CLI: --config --dataset --output-dir --log-level
└── requirements.txt             — all deps including streamlit
```

---

## Pipeline de ejecución (21 checks)

```
config.yaml → load_dataset()
                    │
        ┌───────────┼──────────────────────────────────┐
        ▼           ▼           ▼           ▼           ▼
  quality_checks leakage_checks feature_analysis sufficiency drift_checks
     (6)           (4)            (3)               (4)        (2)
        └───────────┴──────────────────────────────────┘
                    │            │
               impact_analysis  plugins (custom checks)
                    │
              recommendations → ReadinessScore (0-100, A-F)
                    │
              generate_report() → HTML + 4 embedded plots
```

---

## Módulos clave

### `src/utils.py` — Dataclasses compartidos
- `CheckResult`: `check_name, passed, severity, message, details, affected_columns`
- `Recommendation`: `check_name, priority, action, rationale, code_snippet`
- `DimensionScore`: `score, checks_total, checks_passed, errors, warnings`
- `ReadinessScore`: `overall, grade, label, quality, leakage, features, sufficiency`
- `FrameworkReport`: agrega todos los resultados; `all_results()`, `failed_checks()`, `summary()`

### `src/quality_checks.py` — 6 checks
| Check | Método | Severidad |
|-------|--------|-----------|
| `missing_values` | Tasa NaN > threshold | warning/error |
| `duplicates` | Filas exactamente iguales | warning/error |
| `outliers` | IQR o z-score | warning |
| `class_imbalance` | Ratio clase minoritaria | warning/error |
| `constant_features` | nunique() ≤ 1 | warning |
| `low_variance` | CV = std/\|mean\| < threshold | warning |

### `src/leakage_checks.py` — 4 checks
| Check | Método | Severidad |
|-------|--------|-----------|
| `target_leakage` | Pearson \|r\| / Cramér's V ≥ 0.95 | error |
| `train_test_overlap` | Filas duplicadas en split simulado | warning/error |
| `temporal_leakage` | Orden cronológico de date_column | error |
| `id_column_leakage` | Ratio único ≥ 95% en columnas string/int | warning |

**Decisión clave:** `id_column_leakage` excluye float — la alta cardinalidad en variables continuas es normal.

### `src/feature_analysis.py` — 3 checks
| Check | Método |
|-------|--------|
| `feature_correlation` | Pearson \|r\| entre features ≥ 0.90 |
| `feature_relevance` | Mutual information normalizado < 0.01 |
| `distribution_shape` | \|skewness\| > 2.0 o kurtosis > 7.0 |

**Decisión clave:** `low_variance` usa CV en lugar de varianza normalizada (min-max siempre mapea a [0,1], no captura casi-constantes).

### `src/sufficiency.py` — 4 checks (Phase 11)
| Check | Qué valida |
|-------|-----------|
| `sample_size` | n ≥ 100 y ratio n/p ≥ 50 |
| `class_support` | ≥ 30 muestras por clase |
| `cv_stability` | std del CV ≤ 0.10 (lee de impact_results) |
| `feature_to_sample_ratio` | p/n ≤ 0.10 (riesgo overfitting) |

### `src/drift_checks.py` — 2 checks (Phase 14)
- `covariate_drift`: KS test + PSI con binning adaptativo (mín. 20 obs/bin) y corrección de Bonferroni
- `label_drift`: chi-cuadrado sobre distribución del target entre dos mitades

### `src/scoring.py` — ReadinessScore (Phase 10)
```
overall = quality×0.25 + leakage×0.35 + features×0.25 + sufficiency×0.15
  error   → −15 pts (leakage: ×1.5)
  warning → −5 pts
Grado: A≥85, B≥70, C≥55, D≥40, F<40
```

### `src/recommendations.py` — 20 handlers (Phase 8)
Mapea cada check fallido a `Recommendation(priority, action, rationale, code_snippet)`.
Ordenadas por prioridad: high → medium → low.

### `src/plugins.py` — Plugin system (Phase 15)
```python
from src.plugins import register_check
@register_check(phase="quality", name="my_check")
def check_custom(df, target_col, config) -> CheckResult: ...
```
`load_plugins(["my_module.checks"])` en config.yaml.

### `src/checker.py` — Python SDK (Phase 13)
```python
checker = DatasetChecker("configs/config.yaml")
report  = checker.run("data/titanic.csv", target_col="survived")
print(f"{checker.score}/100  {checker.grade}")
checker.save_report("reports/")
d = checker.to_dict()  # JSON-serializable
```

### `app/app.py` — Streamlit (Phase 12)
```bash
streamlit run app/app.py
# → http://localhost:8501
```
Upload CSV/Parquet/Excel, configuración inline, 7 pestañas, descarga HTML+JSON.

---

## Resultados experimentales (21 checks)

| Dataset | Rows | Score | Grade | Pass/Total | Key findings |
|---------|------|-------|-------|-----------|--------------|
| clean_dataset | 500 | ~95 | A | 21/21 | Zero false positives (control) |
| dirty_dataset | 5,300 | ~65 | C | 12/21 | 5 quality issues, 2 sufficiency |
| leaky_dataset | 5,320 | ~72 | B | 10/21 | target_leakage, temporal, ID, features |
| Titanic | 1,309 | ~75 | B | 17/21 | boat column r=0.97, 4 missing cols |
| Diabetes | 768 | ~80 | B | 20/21 | 51 outliers (physiological zeros) |

**Resultado clave:** leaky_dataset baseline accuracy = 1.000 → cleaned = 0.952 (Δ = −0.048).

---

## Ejecución

```bash
# CLI completo
python main.py --config configs/config.yaml --dataset data/raw/titanic.csv

# Python SDK
from src.checker import DatasetChecker
checker = DatasetChecker()
checker.run("data/titanic.csv", target_col="survived")

# Streamlit app
streamlit run app/app.py

# Notebook (kernel ml-framework)
python -m jupyter lab notebooks/framework_demo.ipynb
```

---

## Dependencias principales

```
pandas, numpy, scipy, scikit-learn, xgboost
pyyaml, jinja2, matplotlib, seaborn
loguru, streamlit, pytest, nbformat, jupyter
```
