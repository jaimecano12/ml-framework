# CLAUDE.md — ml-framework (TFM)

## Proyecto

**"An Automated Framework for Dataset Quality Assessment and Data Leakage Detection in Machine Learning"**
TFM de Jaime Cano Moraño — Python 3.13, entorno miniconda (`/opt/miniconda3`).
Repositorio: https://github.com/jaimecano12/ml-framework
Institución: Illinois Institute of Technology

---

## Reglas de trabajo

- **Siempre** hacer `git commit + git push` al terminar cada fase o bloque de trabajo.
- Usar `python -m pip install` y `python -m pytest` (nunca `pip` ni `pytest` a secas: apuntan a Python 3.12, no al entorno activo).
- Para ejecutar el notebook usar el kernel `ml-framework`: `python -m jupyter nbconvert --ExecutePreprocessor.kernel_name=ml-framework`
- Un test roto se arregla antes de seguir; nunca se omite con `--ignore` o `xfail` sin justificación.

---

## Estado actual — COMPLETO (16 fases + notebook + paper)

| Fase | Módulo / Artefacto | Tests | Estado |
|------|--------------------|-------|--------|
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
| 16 — Unified leakage risk score + LLM semantic analysis | `src/leakage_checks.py`, `src/semantic_leakage.py` | 33 | ✅ |
| Demo notebook | `notebooks/framework_demo.ipynb` | — | ✅ |
| Paper | `paper.tex`, `paper.pdf` | — | ✅ |

**Total: 325 tests, 325 passed.**

---

## Arquitectura completa

```
ml-framework/
├── app/
│   └── app.py                      — Streamlit web app (Phase 12)
├── configs/
│   ├── config.yaml                 — Master config (all phases, incl. Phase 16)
│   ├── diabetes_config.yaml
│   ├── leaky_experiment.yaml
│   └── titanic_config.yaml
├── data/raw/                       — 11 datasets (6 synthetic + 5 real-world)
│   ├── clean_dataset.csv           — control (500 rows)
│   ├── dirty_dataset.csv           — quality issues (5,300 rows)
│   ├── leaky_dataset.csv           — leakage issues (5,320 rows)
│   ├── proxy_leakage.csv           — graded noisy proxies (1,000 rows) [Phase 16]
│   ├── temporal_leakage_ext.csv    — churn + future feature (2,000 rows) [Phase 16]
│   ├── multitype_leakage.csv       — ICU proxy+temporal+ID (1,500 rows) [Phase 16]
│   ├── titanic.csv                 — OpenML Titanic (1,309 rows)
│   ├── diabetes.csv                — Pima Diabetes (768 rows)
│   ├── adult.csv                   — Adult Census Income (48,842 rows) [Phase 16]
│   ├── german_credit.csv           — German Credit (1,000 rows) [Phase 16]
│   ├── heart_disease.csv           — Cleveland Heart Disease (303 rows) [Phase 16]
│   └── wine_quality.csv            — Wine Quality Red (1,599 rows) [Phase 16]
├── notebooks/
│   └── framework_demo.ipynb        — 14-cell executed notebook
├── paper.tex                       — LaTeX source (12 pages, conference format)
├── paper.pdf                       — Compiled PDF
├── reports/                        — HTML reports + PNG figures + JSON exports
│   ├── benchmark_results.json      — Tool comparison data [Phase 16]
│   └── benchmark_report.txt        — Human-readable benchmark [Phase 16]
├── scripts/
│   ├── generate_data.py            — 6 synthetic datasets (incl. 3 new [Phase 16])
│   ├── download_real_datasets.py   — Titanic + Diabetes via OpenML
│   ├── download_more_datasets.py   — Adult, Heart Disease, German Credit, Wine [Phase 16]
│   ├── benchmark_comparison.py     — Quantitative benchmark vs 3 tools [Phase 16]
│   ├── run_pipeline.py             — end-to-end demo
│   ├── build_notebook.py           — notebook builder
│   ├── write_section2.py           — writes Section 2 into tfm.docx
│   └── full_evaluation.py          — comprehensive benchmark
├── src/
│   ├── __init__.py                 — public API: DatasetChecker + dataclasses + semantic
│   ├── checker.py                  — DatasetChecker SDK (Phase 13)
│   ├── config.py                   — YAML loader, deep-merge, validation
│   ├── drift_checks.py             — KS + PSI covariate/label drift (Phase 14)
│   ├── feature_analysis.py         — correlation, MI relevance, distribution (Phase 9)
│   ├── impact_analysis.py          — baseline vs cleaned CV comparison (Phase 5)
│   ├── leakage_checks.py           — 5 leakage checks incl. unified risk score (Phase 16)
│   ├── plugins.py                  — @register_check plugin system (Phase 15)
│   ├── quality_checks.py           — 6 quality checks (Phase 3)
│   ├── recommendations.py          — 20 handlers → Recommendation objects (Phase 8)
│   ├── reporting.py                — HTML generation via Jinja2 + matplotlib (Phase 6)
│   ├── scoring.py                  — 0-100 ReadinessScore, A-F grade (Phase 10)
│   ├── semantic_leakage.py         — GPT-4o-mini semantic leakage analysis (Phase 16)
│   ├── sufficiency.py              — 4 statistical sufficiency checks (Phase 11)
│   ├── templates/report.html.j2    — HTML template (inline CSS, no deps)
│   └── utils.py                    — CheckResult, FrameworkReport, Recommendation,
│                                     DimensionScore, ReadinessScore, load_dataset
├── tests/                          — 325 tests across 13 test files
├── main.py                         — CLI: --config --dataset --output-dir --log-level
└── requirements.txt                — all deps including streamlit, openai
```

---

## Pipeline de ejecución (22 checks activos)

```
config.yaml → load_dataset()
                    │
    ┌───────────────┼────────────────────────────────────────────┐
    ▼               ▼               ▼               ▼            ▼
quality_checks  leakage_checks  feature_analysis sufficiency  drift_checks
   (6)             (5)              (3)              (4)          (2)
    └───────────────┴────────────────────────────────────────────┘
                    │                │
               impact_analysis     plugins (custom checks)
                    │
         [opcional] semantic_leakage (GPT-4o-mini via Azure OpenAI)
                    │
              recommendations → ReadinessScore (0-100, A-F)
                    │
              generate_report() → HTML + JSON + 4 embedded plots
```

---

## Módulos clave

### `src/leakage_checks.py` — 5 checks (Phase 4 + Phase 16)
| Check | Método | Severidad |
|-------|--------|-----------|
| `target_leakage` | Pearson \|r\| / Cramér's V ≥ 0.95 | error |
| `train_test_overlap` | Filas duplicadas en split simulado | warning/error |
| `temporal_leakage` | Orden cronológico de date_column | error |
| `id_column_leakage` | Ratio único ≥ 95% en columnas string/int | warning |
| `leakage_risk_score` | Combinación ponderada corr + MI + perf_inflation | warning/error |

**Unified Leakage Risk Score (Phase 16):**
```
L(f) = 0.35·ρ(f) + 0.35·Ĩ(f;y) + 0.30·π(f)
  ρ(f)   = Pearson |r| o Cramér's V ∈ [0,1]
  Ĩ(f;y) = MI normalizado por max MI ∈ [0,1]
  π(f)   = (A_f - A_base) / (1 - A_base) ∈ [0,1]
  Flag: L(f) ≥ 0.7 → warning; ≥ 0.9 → error
```

### `src/semantic_leakage.py` — LLM analysis (Phase 16)
- Envía feature names + sample values + descripción del dataset a GPT-4o-mini via Azure OpenAI
- Devuelve `SemanticRiskAssessment` por feature: risk_level (none/low/medium/high) + leakage_type (temporal/proxy/post_hoc/indirect)
- Requiere `AZURE_OPENAI_API_KEY` y `AZURE_OPENAI_ENDPOINT` en env vars
- Habilitado con `semantic_leakage.enabled: true` en config.yaml (default: false)
- Degrada gracefully si no hay credenciales

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

### `src/scoring.py` — ReadinessScore
```
overall = quality×0.25 + leakage×0.35 + features×0.25 + sufficiency×0.15
  error   → −15 pts
  warning → −5 pts
Grado: A≥85, B≥70, C≥55, D≥40, F<40
```

---

## Resultados experimentales

### Datasets principales (21 checks)
| Dataset | Rows | Score | Grade | Pass/Total | Key findings |
|---------|------|-------|-------|-----------|--------------|
| clean_dataset | 500 | 95 | A | 21/21 | Zero false positives (control) |
| dirty_dataset | 5,300 | 65 | C | 12/21 | 5 quality issues, 2 sufficiency |
| leaky_dataset | 5,320 | 72 | B | 11/21 | target_leakage, temporal, ID |
| Titanic | 1,309 | 75 | B | 17/21 | boat column r=0.97 |
| Diabetes | 768 | 80 | B | 20/21 | 51 outliers (physiological zeros) |

### Datasets UCI adicionales (Phase 16)
| Dataset | Rows | Score | Grade | Primary issues |
|---------|------|-------|-------|----------------|
| Adult Census | 48,842 | 68 | C | Class imbalance (76/24), missing values |
| German Credit | 1,000 | 74 | B | Class imbalance (70/30), feature correlation |
| Heart Disease | 303 | 62 | C | Low n/p=21.6, inter-feature correlation |
| Wine Quality | 1,599 | 82 | B | Mild imbalance, skewness in residual.sugar |

**Resultado clave:** leaky_dataset baseline accuracy = 1.000 → cleaned = 0.952 (Δ = −0.048).

### Benchmark vs herramientas (Phase 16)
| Herramienta | Checks (de 29) | Detección leakage (de 4) |
|-------------|---------------|--------------------------|
| ml-framework | **29/29** | **4/4** |
| Deepchecks | 11/29 | 0/4 |
| Great Expectations | 9/29 | 1/4 |
| ydata-profiling | 8/29 | 0/4 |

---

## Paper académico

- **Archivo:** `paper.tex` / `paper.pdf`
- **Formato:** 12 páginas, single-column, 11pt Times New Roman, estilo conferencia
- **Compilar:** `tectonic paper.tex` (requiere Homebrew `tectonic`)
- **Referencias:** 19 (Kaufman 2012, Sculley 2015, Breck 2019, Pedregosa 2011, Chen 2016, Kraskov 2004, Ross 2014, Rabanser 2019, Siddiqi 2006, McKinney 2010, OpenAI 2023, UCI 2017, Zha 2023, Narayan 2022, Sui 2023, Ng 2021, ydata-profiling, Great Expectations, Deepchecks)
- **Formato técnico:** fancyhdr (header/footer), mdframed (abstract box), titlesec (línea bajo secciones), captionsetup (labels en negrita), arraystretch=1.12, listings con fondo gris, widowpenalty/clubpenalty, emergencystretch

### Estructura del paper y dónde está cada contribución

| Sección | Páginas | Contenido clave |
|---------|---------|-----------------|
| Abstract | 1 | Resumen: 4/4 leakage detection, 29 checks, benchmark |
| 1. Introduction | 1–2 | Motivación, 3 ejemplos reales (Titanic boat, Δ=-0.048, ICU semántico), 4 contribuciones |
| 2. Related Work | 2–3 | Leakage, data quality tools, data-centric AI, LLMs, MI, drift |
| 3. System Architecture | 3–5 | Pipeline TikZ, interfaces (CLI/SDK/Streamlit), listing SDK, tabla módulos |
| 4. Core Methodology | 5–8 | Quality checks, 4 leakage checks clásicos, **unified risk score (Eq.1 + Algorithm 1)**, feature analysis, sufficiency, drift (Eq.2), impact analysis, readiness score (Eq.3), **LLM semántico** |
| 5. Experimental Evaluation | 8–10 | 11 datasets, readiness scores, **tabla UCI extendida**, impact analysis (Δ=-0.048), validación L(f) |
| 6. Quantitative Benchmark | 10–11 | **29/29 vs 8–11**, **4/4 detection vs 0–1**, tabla flexibilidad, runtime |
| 7. Discussion | 11–12 | Fortalezas, complementariedad de señales, limitaciones, 4 direcciones futuras |
| 8. Conclusion | 12 | Resultados clave + GitHub |

### Mapa feedback Prof. Yong → sección del paper

| Feedback | Implementado en | Sección paper |
|----------|----------------|---------------|
| Unified leakage risk score | `leakage_checks.py::check_leakage_risk_score()` | §4.2.2, Eq.1, Algorithm 1, Table 4 |
| Más datasets + escenarios complejos | `data/raw/` (11 datasets), `generate_data.py` | §5.1, §5.2, Table 3 |
| Benchmark cuantitativo vs herramientas | `scripts/benchmark_comparison.py` | §6, Fig.3, Tables 5–7 |
| LLM semantic analysis | `src/semantic_leakage.py` | §4.7, §7 (Discussion) |
| Reducir detalles de implementación | Paper escrito como research paper puro | Todo el paper |

---

## Ejecución

```bash
# CLI completo
python main.py --config configs/config.yaml --dataset data/raw/titanic.csv

# Python SDK
from src.checker import DatasetChecker
checker = DatasetChecker("configs/config.yaml")
report = checker.run("data/titanic.csv", target_col="survived")
print(f"{checker.score}/100  grade={checker.grade}")
checker.save_report("reports/")

# Streamlit app
streamlit run app/app.py

# Generar datasets sintéticos (incl. 3 nuevos)
python scripts/generate_data.py

# Descargar datasets UCI adicionales
python scripts/download_more_datasets.py

# Benchmark vs otras herramientas
python scripts/benchmark_comparison.py

# Compilar paper
tectonic paper.tex

# Notebook (kernel ml-framework)
python -m jupyter lab notebooks/framework_demo.ipynb
```

---

## Dependencias principales

```
pandas, numpy, scipy, scikit-learn, xgboost
pyyaml, jinja2, matplotlib, seaborn
loguru, streamlit, pytest, nbformat, jupyter
openai                  # LLM semantic analysis (opcional)
ydata-profiling         # benchmark comparison
deepchecks              # benchmark comparison
great-expectations      # benchmark comparison
tectonic                # compilar paper (brew install tectonic)
```

---

## Feedback del supervisor (Prof. Yong)

Todo incorporado en Phase 16 y documentado en el paper con secciones específicas:

1. ✅ **Unified leakage risk score** → `src/leakage_checks.py` → paper §4.2.2, Eq.1, Algorithm 1, Table 4
2. ✅ **Más datasets + escenarios complejos** → `data/raw/` (11 datasets) → paper §5.1–5.2, Table 3
3. ✅ **Benchmark cuantitativo** → `scripts/benchmark_comparison.py` → paper §6, Fig.3, Tables 5–7
4. ✅ **LLM semantic analysis** → `src/semantic_leakage.py` → paper §4.7, §7
5. ✅ **Paper sin detalles de implementación** → `paper.tex` (12 páginas, research-oriented)

**Respuesta enviada al profesor** indicando sección exacta del paper para cada punto (mail redactado 2026-06-05).
