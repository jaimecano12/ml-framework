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

## Estado actual — COMPLETO (17 fases + notebook + paper)

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
| 17 — LLM benchmark + configurable scoring + case studies | `data/semantic_benchmark.json`, `scripts/evaluate_semantic_leakage.py`, `scripts/case_studies.py`, `src/scoring.py`, `configs/config.yaml` | — | ✅ |
| Demo notebook | `notebooks/framework_demo.ipynb` | — | ✅ |
| Paper | `paper.tex`, `paper.pdf` (15 pages) | — | ✅ |

**Total: 343 tests, 343 passed** (325 al cierre de Fase 17; +6 en la integración de AWS Bedrock;
+9 al conectar el módulo semántico al pipeline real — ver secciones correspondientes más abajo).

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
├── paper.tex                       — LaTeX source (15 pages, conference format)
├── paper.pdf                       — Compiled PDF
├── reports/                        — HTML reports + PNG figures + JSON exports
│   ├── benchmark_results.json      — Tool comparison data [Phase 16]
│   └── benchmark_report.txt        — Human-readable benchmark [Phase 16]
├── scripts/
│   ├── generate_data.py            — 6 synthetic datasets (incl. 3 new [Phase 16])
│   ├── download_real_datasets.py   — Titanic + Diabetes via OpenML
│   ├── download_more_datasets.py   — Adult, Heart Disease, German Credit, Wine [Phase 16]
│   ├── benchmark_comparison.py     — Quantitative benchmark vs 3 tools [Phase 16]
│   ├── evaluate_semantic_leakage.py — P/R/F1 evaluation of LLM module on 30-feature benchmark [Phase 17]
│   ├── case_studies.py             — Runs pipeline on Titanic, Adult, German Credit [Phase 17]
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
│   ├── semantic_leakage.py         — LLM semantic leakage analysis, provider-agnostic:
│   │                                 GPT-4o-mini (Azure) or Claude Haiku 4.5 (AWS Bedrock)
│   ├── sufficiency.py              — 4 statistical sufficiency checks (Phase 11)
│   ├── templates/report.html.j2    — HTML template (inline CSS, no deps)
│   └── utils.py                    — CheckResult, FrameworkReport, Recommendation,
│                                     DimensionScore, ReadinessScore, load_dataset
├── tests/                          — 343 tests across 13 test files
├── main.py                         — CLI: --config --dataset --output-dir --log-level
└── requirements.txt                — all deps including streamlit, openai
                                       (boto3 for AWS Bedrock: optional extra, not pinned here)
```

---

## Pipeline de ejecución (20 checks activos + impact analysis + semantic opcional)

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
         [opcional] semantic_leakage (GPT-4o-mini/Azure o Claude Haiku 4.5/AWS Bedrock)
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

### `src/semantic_leakage.py` — LLM analysis (Phase 16, provider-agnostic desde 2026-07-29)
- Envía feature names + sample values + descripción del dataset a un LLM; dos proveedores intercambiables via `config.yaml::semantic_leakage.provider`:
  - `azure` (default histórico): GPT-4o-mini, requiere `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`. **Nunca se obtuvieron credenciales para este proyecto.**
  - `bedrock` (default actual en config.yaml): Claude Haiku 4.5 vía AWS `boto3` bedrock-runtime (`converse()`), requiere credenciales AWS (`aws configure`) + `AWS_REGION`. Model id: `us.anthropic.claude-haiku-4-5-20251001-v1:0` — **ojo:** este modelo concreto exige un *cross-region inference profile* (prefijo `us.`), no el model id "pelado"; si Bedrock devuelve error de on-demand throughput, comprobar con `aws bedrock list-inference-profiles`.
- Dispatcher interno: `_call_llm(prompt, model_id, provider=...)` → `_call_llm_azure()` o `_call_llm_bedrock()`. Los tests mockean `_call_llm` directamente, así que ambos providers están cubiertos sin romper los tests existentes.
- Devuelve `SemanticRiskAssessment` por feature: risk_level (none/low/medium/high) + leakage_type (temporal/proxy/post_hoc/indirect)
- Habilitado con `semantic_leakage.enabled: true` en config.yaml (default: false)
- Degrada gracefully si no hay credenciales (`EnvironmentError` → `CheckResult` passing con mensaje "skipped")
- **Conectado al pipeline real desde 2026-07-30** (antes era una función suelta que nadie llamaba) —
  ver sección "Conexión del módulo semántico al pipeline real" más abajo.

**Evaluación en vivo (2026-07-29, `scripts/evaluate_semantic_leakage.py --provider bedrock`)** sobre
`data/semantic_benchmark.json` (30 features) — sustituye el resultado mock anterior (P=1.00/R=0.93/F1=0.963):

| Umbral | P | R | F1 | TP | FP | FN |
|--------|---|---|----|----|----|----|
| `medium` (default) | 1.000 | 1.000 | 1.000 | 14 | 0 | 0 |
| `high` | 0.917 | 0.846 | 0.880 | 11 | 1 | 2 |

3 discrepancias reales en `high` (evidencia genuina del modelo, no artefacto del mock):
`treatment_count` (gt=medium, pred=high — ya documentada como ambigua), `batch_rejection_rate`
y `training_completion_post_hire` (gt=high, pred=medium — infravaloradas). Propagado a
`tfm.tex`, `paper.tex` y la presentación — ver sección dedicada más abajo.

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

**⚠️ Estos son los números vigentes (re-ejecutados sobre el pipeline actual, 2026-06-29).
Los valores anteriores de esta tabla estaban obsoletos/no reproducibles y fueron sustituidos
tras la segunda ronda de revisión — ver sección de feedback más abajo.**

### Datasets principales (20 checks + impact analysis)
| Dataset | Rows | Score | Grade | Pass/Total | Key findings |
|---------|------|-------|-------|-----------|--------------|
| clean_dataset | 500 | 98.8 | A | 21/23 | Near-zero MI en 1 feature; drift leve |
| dirty_dataset | 5,300 | 91.2 | A | 16/23 | 7 warnings calidad/features, sin leakage |
| leaky_dataset | 5,320 | 71.6 | B | 12/23 | target_leakage + LRS errors |
| Titanic | 1,309 | 80.6 | B | 14/22 | `name` (Cramér's V=0.999); `boat` LRS=0.74 |
| Diabetes | 768 | 96.2 | A | 19/22 | 51 outliers en 4 cols |

### Datasets UCI adicionales (Phase 16)
| Dataset | Rows | Score | Grade | Primary issues |
|---------|------|-------|-------|----------------|
| Adult Census | 48,842 | 92.4 | A | Missing values, 52 duplicates, skew en capital-gain |
| German Credit | 1,000 | 97.5 | A | 3 features con MI casi nula, outliers |
| Heart Disease | 303 | 96.8 | A | Low n/p=23.3 (target corregido, ver nota abajo) |
| Wine Quality | 1,599 | 88.6 | A | 240 duplicates (15%, error), drift en 9/11 features |

**Resultado clave:** leaky_dataset (LR) baseline accuracy = 1.000 → cleaned = 0.952 (Δ = −0.048).

**Nota Heart Disease:** `scripts/download_more_datasets.py::download_heart_disease()` tenía un
bug real (regex que colapsaba el target a una sola clase); corregido — ver sección de feedback.

### Benchmark vs herramientas (Phase 16)
"29" = checklist de comparación cross-tool (más granular que el pipeline propio de 20 checks,
no confundir ambos números — ver `scripts/benchmark_comparison.py::FEATURE_MATRIX`).

| Herramienta | Checklist items (de 29) | Detección leakage (de 4) |
|-------------|---------------|--------------------------|
| ml-framework | **29/29** | **4/4** |
| Deepchecks | 11/29 | 0/4 |
| Great Expectations | 9/29 | 1/4 |
| ydata-profiling | 8/29 | 0/4 |

---

## Paper académico

- **Archivo:** `paper.tex` / `paper.pdf`
- **Formato:** 15 páginas, single-column, 11pt Times New Roman, estilo conferencia
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
| 4. Core Methodology | 5–9 | Quality checks, 4 leakage checks clásicos, **unified risk score (Eq.1 + Algorithm 1)**, feature analysis, sufficiency, drift (Eq.2), impact analysis, readiness score (Eq.3 + **weight rationale**), **LLM semántico + Tabla evaluación P/R/F1** |
| 5. Experimental Evaluation | 9–11 | 11 datasets, readiness scores, tabla UCI extendida, impact analysis, validación L(f), **§5.5 real-world case studies (Titanic, Adult, German Credit)** |
| 6. Quantitative Benchmark | 11–12 | 29/29 vs 8–11, 4/4 detection vs 0–1, tabla flexibilidad, runtime, **párrafo fairness del benchmark** |
| 7. Discussion | 12–13 | Fortalezas, complementariedad, limitaciones, 4 direcciones futuras |
| 8. Conclusion | 13 | Resultados clave + GitHub |

### Mapa feedback Prof. Yong (segunda ronda) → sección del paper

| Feedback | Implementado en | Sección paper |
|----------|----------------|---------------|
| Unified leakage risk score | `leakage_checks.py::check_leakage_risk_score()` | §4.2.2, Eq.1, Algorithm 1, Table 4 |
| Más datasets + escenarios complejos | `data/raw/` (11 datasets), `generate_data.py` | §5.1, §5.2, Table 3 |
| Benchmark cuantitativo vs herramientas | `scripts/benchmark_comparison.py` | §6, Fig.3, Tables 5–7 |
| LLM semantic analysis | `src/semantic_leakage.py` | §4.7, §7 (Discussion) |
| **LLM quantitative evaluation** | `data/semantic_benchmark.json`, `scripts/evaluate_semantic_leakage.py` | §4.7 Table 8 — evaluación en vivo (Claude Haiku 4.5/Bedrock, 2026-07-29): P=1.00/R=1.00/F1=1.00 (medium), P=0.917/R=0.846/F1=0.880 (high); ya no es un resultado mock |
| **Real-world case studies** | `scripts/case_studies.py`, `reports/case_studies.json` | §5.5, Table 9 |
| **Readiness score justification** | `src/scoring.py`, `configs/config.yaml` (scoring block) | §4.6 weight rationale paragraph |
| **Benchmark fairness discussion** | Paper §6.2 | Párrafo "Scope of comparison" |
| **Configurable parameters** | `configs/config.yaml` scoring block, `src/scoring.py` | §4.6 + config.yaml |

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

# Evaluación cuantitativa del módulo semántico
python scripts/evaluate_semantic_leakage.py --provider mock      # sin API key
python scripts/evaluate_semantic_leakage.py --provider azure     # GPT-4o-mini (nunca configurado)
python scripts/evaluate_semantic_leakage.py --provider bedrock   # Claude Haiku 4.5 (usado en la tesis)

# Case studies reales (Titanic, Adult, German Credit)
python scripts/case_studies.py

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
openai                  # LLM semantic analysis, provider=azure (opcional, nunca usado con creds reales)
boto3                   # LLM semantic analysis, provider=bedrock (opcional; el que sí se usa)
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

### Segunda ronda de revisión — issues críticos (2026-06-29)

Feedback recibido: números inconsistentes (29 vs 21 checks), readiness scores contradictorios
entre tablas (Titanic/Adult/German Credit con valores distintos en Tabla 3 vs Tabla 7), cita
rota `[?]` en página 9, Heart Disease ausente del conteo de datasets en §5.1. Todo corregido
en commit `6323ffb`:

1. ✅ **Conteo de checks** → estandarizado: pipeline propio = **20 checks** (6+5+3+4+2) en 5
   dimensiones; el **29** pasa a describirse explícitamente como un *checklist* de comparación
   cross-tool separado (`scripts/benchmark_comparison.py::FEATURE_MATRIX`), usado solo en §6.
2. ✅ **Scores contradictorios** → causa real encontrada: un bug de datos, no solo de
   redacción. `scripts/download_more_datasets.py::download_heart_disease()` derivaba el target
   con una regex que extraía dígitos de las etiquetas categóricas `'<50'`/`'>50_1'` — ambas
   contienen "50", colapsando el target a una sola clase en las 303 filas. Corregido derivando
   el label directamente del string de categoría (split real 165/138). Se re-ejecutó el pipeline
   completo sobre los 9 datasets reales/sintéticos relevantes y se reemplazaron todos los
   números de las Tablas 3/4/6/7 y Fig.2 del paper con un único conjunto de resultados
   reproducible (`tectonic paper.tex` + pytest 325/325 verificado tras el fix).
3. ✅ **Cita rota** → `\cite{siddiqi2006}` (sin definir) → `\cite{siddiqi2006credit}`; además se
   añadieron 6 `\cite` que faltaban para entradas de la bibliografía nunca citadas
   (chen2016xgboost, dua2017uci, mckinney2010data, ng2021datacentric, openai2023gpt4,
   pedregosa2011scikit).
4. ✅ **Heart Disease ausente en §5.1** → abstract y §5.1 ahora dicen "six real-world + six
   synthetic (twelve total)", listando Heart Disease explícitamente.
5. ✅ **Pesos LRS sin justificar** → Tabla 8 (ablation): bajo un esquema *correlation-heavy*,
   `boat` de Titanic cae de $\mathcal{L}=0.74$ a $0.66$ (por debajo del umbral) — evidencia
   empírica de por qué los pesos por defecto no sobreponderan la correlación.
6. ✅ **Benchmark semántico "demasiado perfecto"** → se descubrió que la Tabla de evaluación
   semántica se generó con `--mock` (un emparejador de patrones que conoce los nombres exactos
   del benchmark), no con GPT-4o-mini real. Sin credenciales de Azure disponibles, se optó
   (decisión del usuario) por declararlo explícitamente como validación del arnés de evaluación,
   no como medida de precisión real del LLM; evaluación en vivo movida a future work prioritario.
7. ✅ **"Outperforms" demasiado fuerte** → abstract reescrito a "complements ... fills a gap".

✅ **Propagado a `tfm.tex`** (commit `4602b7e`): mismo trabajo replicado en el capítulo
Results/Conclusions — incluye un bug propio adicional encontrado ahí (`tab:all_checks`
afirmaba "22 total" pero la tabla solo listaba 21 filas), la tabla de ablation de pesos LRS,
la divulgación del mock semántico, y la corrección de Heart Disease + verificación empírica
de que el bug de datos NO explica el antiguo número 62/100 (el dataset roto de una sola
clase en realidad puntúa 94.2/A con el pipeline actual, así que ese número se trata como
no reproducible/superado, no como causado por el bug).

### Cierre del caveat semántico: integración de AWS Bedrock (2026-07-29)

El punto 6 de arriba (benchmark semántico "demasiado perfecto") quedó resuelto de raíz:
el usuario tenía créditos de AWS (nunca credenciales de Azure), así que en vez de esperar
a conseguir acceso a Azure se integró **AWS Bedrock** como segundo proveedor del módulo LLM.

1. ✅ **Código** (`src/semantic_leakage.py`): dispatcher `provider="azure"|"bedrock"`;
   `_call_llm_bedrock()` usa `boto3` (`bedrock-runtime.converse()`); Azure intacto. Modelo:
   Claude Haiku 4.5, id `us.anthropic.claude-haiku-4-5-20251001-v1:0` (requiere *inference
   profile* con prefijo `us.`, no el model id pelado — gotcha real de Bedrock para modelos
   recientes). 6 tests nuevos, 325→331 totales.
2. ✅ **Evaluación real** (`scripts/evaluate_semantic_leakage.py --provider bedrock`) contra
   las 30 features del benchmark: P/R/F1 = 1.00/1.00/1.00 en `medium` (umbral operativo),
   0.917/0.846/0.880 en `high`, con 3 discrepancias explicables (ver detalle en la sección
   de `semantic_leakage.py` más arriba). Ya no es necesario declarar el resultado como
   "solo validación del arnés" — es un modelo real, independiente, evaluado en vivo.
3. ✅ **Propagado a `tfm.tex` y `paper.tex`**: abstract, metodología, la sección completa de
   evaluación semántica (reescrita con tabla de ambos umbrales + tabla de discrepancias),
   discusión, limitaciones, y Future Work (el ítem "live semantic-module evaluation" se
   quitó por estar hecho; el future work de "larger benchmark" ahora es explícitamente
   multi-provider). Conteo de tests 325→331 corregido en todas las tablas/menciones,
   incluida una inconsistencia aritmética preexistente en la tabla de distribución de tests
   de `tfm.tex` (no causada por este cambio, pero corregida de paso para las cifras que sí
   tocaba: fila `test_semantic_leakage.py` y el total).
4. ✅ **Propagado a la presentación** — ver sección dedicada más abajo.

### Conexión del módulo semántico al pipeline real (2026-07-30)

Hasta este punto `analyse_semantic_leakage()` era una función que nadie llamaba desde el
pipeline real: `checker.py::run()`, `main.py` y `app/app.py` nunca la invocaban, así que poner
`semantic_leakage.enabled: true` en `config.yaml` no tenía ningún efecto, y el resultado no
aparecía ni en el HTML ni en `to_dict()` ni en `FrameworkReport`. Conectado de extremo a extremo:

1. ✅ **`src/config.py`**: `semantic_leakage` ahora tiene defaults reales en `_DEFAULTS`
   (disabled por defecto) + validación de `provider`/`risk_threshold`, así que
   `DatasetChecker()` sin config file también funciona.
2. ✅ **`src/utils.py`**: `FrameworkReport.semantic_results` (incluido en `all_results()` /
   `failed_checks()` / `summary()`).
3. ✅ **`src/checker.py`**: `run()` llama a `analyse_semantic_leakage()` si
   `semantic_leakage.enabled` es true; nuevo parámetro opcional `dataset_description`;
   `to_dict()` lo exporta.
4. ✅ **`src/reporting.py` + `report.html.j2`**: nueva sección "Semantic Leakage Analysis (LLM)"
   en el informe HTML — **solo lista features con `risk_level != "none"`** (antes se listaban
   las 12 features de Titanic con 11 filas irrelevantes; ahora solo `boat`), con contador
   "N of M feature(s) flagged".
5. ✅ **`app/app.py`**: checkbox en sidebar para activarlo + selector de provider (bedrock/azure)
   + campo de descripción, y pestaña nueva **"🤖 Semantic"**. Rediseñada dos veces:
   - v1: `st.dataframe` plano (poco atractivo, texto cortado, la única fila relevante enterrada
     entre 11 "none").
   - v2 (definitiva): métricas arriba (Status/Features flagged/Severity) + filtro de radio
     (All/High/Medium/Low, **sin "None"**) + un `st.expander` por feature con badges de color,
     igual que la pestaña "Recommendations" — y **solo se listan las features flagged**, con un
     caption indicando cuántas adicionales salieron limpias.
6. ✅ **9 tests nuevos** (config validation/defaults, checker con `_call_llm` mockeado,
   agregación en `FrameworkReport`, HTML con/sin la sección, HTML omite `risk=none`).
   331→340→343 tests totales.

**Verificado con una llamada real a Bedrock** (no mockeada) sobre Titanic vía
`DatasetChecker` directamente y vía la app de Streamlit real (Playwright + Chromium headless,
`python -m pip install playwright && playwright install chromium`): el modelo marca `boat`
como `high`/`post_hoc` de forma independiente — confirmación cruzada del hallazgo estadístico
de la tesis. Screenshot de verificación tomado y revisado antes de dar el trabajo por bueno
(no se instaló `chromium-cli`, que es el método preferido de la skill `run` pero no estaba
disponible; Playwright fue el fallback documentado en la propia skill).

---

## Revisión del Prof. Yong Zheng + estrategia de publicación (2026-08-12)

El profesor (tutor real: Yong Zheng — "Prof. Yong" en el resto de este fichero, "Prof. Zheng" en
la correspondencia) revisó `tfm.tex` y `paper.tex` juntos y mandó una lista de errores que había
que arreglar antes de cualquier submission, más dos opciones de publicación: **Opción A** (poster
de 2 páginas, ACM SIGCITE, deadline el domingo) y **Opción B** (extender el trabajo ~2-3 meses
para journal: SoftwareX / Software Impacts / DMLR). Su recomendación: hacer las dos, en secuencia.

Antes de proponer nada se verificó cada punto contra el código real (no se dio nada por
supuesto) — y aparecieron **dos contradicciones numéricas más que el profesor no había visto**.
Se decidió seguir su recomendación (A → B) y arreglar todo lo de la Parte 1 de inmediato.

### El bug de Deepchecks (el más serio) — causa raíz encontrada y arreglada

`scripts/benchmark_comparison.py::_detect_deepchecks()` reportaba **0/4** escenarios de leakage
detectados por Deepchecks (Tabla `tab:detection`), mientras que `run_runtime_benchmark()`
reportaba que Deepchecks ni siquiera podía ejecutarse ("—", error de compatibilidad NumPy 2.0)
— contradicción interna real que el profesor señaló. Investigado y confirmado: eran **dos bugs
distintos**, no una detección genuinamente fallida:

1. `deepchecks.tabular.checks` no importa bajo NumPy ≥ 2.0 (`np.Inf` fue eliminado de NumPy en
   la versión 2.0; el código interno de Deepchecks 0.19.1 — `performance_bias.py` — todavía lo
   usa). El `try/except Exception` de `_detect_deepchecks()` convertía ese `AttributeError` en
   `"detected": False`, confundiendo "no pudo ejecutarse" con "no detectó nada".
2. Incluso arreglado el entorno, el código original pasaba `FeatureLabelCorrelation(ppscore_threshold=0.8)`
   — `ppscore_threshold` **no es un parámetro válido** de esa clase (el real es `ppscore_params`;
   el constructor acepta `**kwargs` así que el error pasaba desapercibido). Nunca se añadía una
   condición real ni se leía `result.value`, así que `detected` era `False` sin importar los
   datos.

**Arreglo**: entorno conda aislado `deepchecks_compat` (`python=3.11`, `numpy<2` vía
`pip install "numpy<2" "scikit-learn==1.4.2" "category-encoders==2.6.3" deepchecks pandas scipy`
— ojo, hubo que fijar también `scikit-learn` y `category-encoders`, no solo numpy, porque las
versiones más recientes de esos dos rompen el import de Deepchecks por otras dos razones
distintas: `sklearn.metrics.get_scorer('max_error')` y `sklearn.utils.Tags` respectivamente).
`_detect_deepchecks()` reescrito en `scripts/benchmark_comparison.py`: usa
`IdentifierLabelCorrelation` (con la columna ID declarada como `index_name` del `Dataset`, no
como feature genérica) para el escenario `id_column`, y `FeatureLabelCorrelation` con
`add_condition_feature_pps_less_than()` + lectura de `result.value` para los otros tres.

**Resultado real, verificado en el entorno aislado: Deepchecks detecta 3/4** (falla solo
`id_column` — y por una razón principled, no un fallo: `IdentifierLabelCorrelation` mide
correlación identificador↔target, y ese escenario sintético tiene un ID aleatorio con
correlación cero *por construcción*; es un target de detección distinto — riesgo de
memorización por cardinalidad — al que sí cubre `id_column_leakage` de ml-framework). Runtime de
Deepchecks también verificado real en el mismo entorno: 0.44s / 0.42s / 3.40s (500/1000/5300
filas) — ya no hay "—".

Esto obligó a revisar **todas** las afirmaciones "4×"/"0/4"/"mejor alternativa" en ambos
documentos (Abstract, Introducción, Tabla `tab:detection`, Tabla `tab:runtime`, "Scope of
comparison", Discussion, Conclusion) — con Deepchecks en 3/4, ml-framework (4/4) ya no es "4×
mejor", es "el único con 4/4, frente a 3/4 del más cercano". Se reescribió cada mención para que
la ventaja reportada sea la real (más estrecha, pero defendible — exactamente lo que pedía el
profesor: *"an honest, narrower advantage is far more publishable"*).

### Otros arreglos de la Parte 1

- ✅ **Contradicción `boat`**: `paper.tex` (Introducción) afirmaba `Pearson |r|=0.97` para `boat`;
  la Tabla 7 del mismo paper decía `ρ=0.420`. La cifra 0.420 es la correcta (la de la que depende
  todo el argumento de complementariedad). Reescrito el ejemplo de la introducción para contar la
  historia correcta: correlación baja (0.420, por debajo del umbral 0.95) que un check de
  correlación se perdería, pero que el score unificado sí atrapa vía MI + performance inflation.
- ✅ **Segunda contradicción, no vista por el profesor**: `tfm.tex` (Resumen de contribuciones,
  §5.1.1) afirmaba `L(f)≥0.83` para toda feature con leakage, pero la Tabla 4.5 que cita muestra
  `boat=0.739`. Corregido a `≥0.739`, con matiz añadido de que el margen no es uniforme (los
  proxies sintéticos están en ≥0.99, `boat` mucho más cerca del umbral 0.7 precisamente por su
  correlación débil).
- ✅ **Bibliografía en `paper.tex`** (verificado autor real de `narayan2022can` vía arXiv:2205.09911):
  `rabanser2019failing` "I. Steinwart"→**Z. Lipton**; `zha2023datacentric` "K. Zha"→**D. Zha**;
  `ydataprofiling` arXiv no relacionado→cita del repo (como ya hacía `tfm.tex`);
  `narayan2022can`/`narayan2022` autor list incorrecto en **ambos** documentos→corregido a
  Avanika Narayan, Ines Chami, Laurel Orr, Simran Arora, Christopher Ré en los dos.
- ✅ **Heart Disease**: verificado que `paper.tex` **ya** cumplía lo que pedía el profesor (solo
  describe la corrección del encoding, sin mencionar la cifra irreproducible 62/100) — cero
  trabajo adicional necesario ahí; la discusión más extensa vive solo en `tfm.tex`, donde es
  apropiada como registro interno.
- ✅ **Checklist de 29 ítems**: reordenado en `paper.tex` (Abstract + bullet de contribuciones de
  la Introducción) para que la Detection Rate cuantitativa (4/4 vs 3/4) sea el argumento
  principal y el checklist quede explícitamente como "supplementary, qualitative capability
  audit" — sin mover la Sección 6 a un apéndice físico (eso queda pendiente para si se hace la
  Opción B/journal, ya que el póster de 2 páginas no va a incluir el checklist de todos modos).
- ✅ **Framing del F1=1.00 semántico**: verificado que `paper.tex` ya solo repite esa cifra en la
  Sección 4.6 (ya con el matiz de "strong result... not zero-error behaviour in general") — no
  aparece sin contexto en Abstract/Intro/Conclusion, así que no hacía falta ningún cambio ahí.

Ambos documentos recompilados (`tectonic`, 0 errores) y verificados visualmente
(Abstract, Tabla `tab:detection`, página del bug de Deepchecks) tras cada tanda de cambios.

**Pendiente de decisión del usuario**: el email de respuesta al profesor ya se envió confirmando
el plazo del viernes para Deepchecks (con la causa raíz ya resuelta, no es una incógnita). Falta
por decidir/comunicar cuánto tiempo puede comprometer para la Opción B (2-3 meses que pide el
profesor probablemente optimista; estimación propia: 4-6 meses reales para los 5 bloques (a)-(e)
que describe).

### Construcción del póster SIGCITE de 2 páginas (2026-08-13)

El profesor respondió tras ver el primer borrador (ver más abajo): el póster **encaja en SIGCITE
solo si se reformula el tema de "data leakage and quality" como una cuestión de seguridad**; y
recomendó dejar solo el hallazgo más fuerte en el póster (1-5 de un hipotético 1-8), reservando
el resto para la extensión a journal — lo cual, en la práctica, ya se cumplía solo: el póster usa
exclusivamente resultados que ya existían en `paper.tex` (caso `boat` + ablation de pesos), nada
del trabajo nuevo de la Opción B.

**Primer intento — Word real del profesor:**
El profesor adjuntó por email `acm_submission_template.docx` (la plantilla oficial genérica de
envío de ACM, no algo específico de SIGCITE) — el usuario la tenía en `~/Downloads/` y se localizó
ahí (`ls -t ~/Downloads/*.docx`). Inspeccionada con `python-docx`: es un documento de instrucciones
de ~194 párrafos con todos los estilos ACM (`Title_document`, `Authors`, `Abstract`,
`CCSDescription`, `KeyWords`, `Head1`, `PostHeadPara`, `Para`, `TableCaption`, `FigureCaption`,
`Bib_entry`, etc.) y texto de relleno/Lorem ipsum a sustituir. Confirmado en el propio texto de la
plantilla: **la versión de envío es a una sola columna** ("It should remain in a one-column
format—please do not alter any of the styles or margins"; ACM la convierte a 2 columnas tras la
aceptación) — así que el límite real de "2 páginas" se aplica al resultado final en 2 columnas, no
al borrador de revisión en 1 columna (que salió en 4 páginas con el mismo contenido que luego
ocupó exactamente 2 páginas en `acmart`/`sigconf`, confirmando la equivalencia 1:2).

Reconstruido un nuevo `.docx` con `python-docx` reutilizando los estilos exactos de la plantilla
(se vacía el `body` del documento original y se reinserta contenido nuevo con los mismos nombres
de estilo, en vez de partir de cero, para no perder la definición de estilos ACM). Dos bugs reales
encontrados al verificar el resultado exportando a PDF vía AppleScript+Word
(`osascript ... open POSIX file ... ; save as active document file format format PDF`):
1. **Numeración duplicada** ("1 1 MOTIVATION", "[1] [1] Sara Kaufman...") — los estilos `Head1` y
   `Bib_entry` de la plantilla ya llevan numeración automática de Word (`numPr` en el XML del
   estilo); el texto añadía el número a mano encima. Arreglado quitando los prefijos manuales.
2. **Etiqueta de la gráfica solapada con una barra** — reposicionada.
Gotcha de la automatización con AppleScript: `set theDoc to open POSIX file "..."` no captura una
referencia usable en Word (error "La variable theDoc no está definida"); hay que hacer `open POSIX
file "..."` como sentencia suelta y luego referirse a `active document` en el siguiente comando,
con un `delay` de por medio.

**Giro final — el usuario pidió PDF en vez de Word, y sin Zheng como coautor:**
Se descartó la ruta Word y se retomó el `.tex` de `acmart`/`sigconf` ya existente
(`poster/sigcite_poster.tex`), aplicando el mismo contenido con el ángulo de seguridad:
- Título: "Data Leakage as a Security Blind Spot: A Unified Risk Score Beyond Correlation".
- `\ccsdesc`/`\keywords` nativos de `acmart` (Security and privacy → Software security
  engineering / Systems security; Computing methodologies → Machine learning).
- Caja "Key insight" (`mdframed`) en acento granate justo bajo el título.
- Tablas con cabecera granate + fila de `boat`/"miss" resaltada; gráfica de barras con verde
  azulado (pasa el umbral) vs. rojo (falla) en vez de un único color.
- Un solo autor (Jaime Cano Moraño) — Yong Zheng retirado a petición explícita del usuario.
Bug real durante la migración: `\usepackage[table]{xcolor}` choca con el xcolor que ya carga
`acmart` internamente ("Option clash for package xcolor") — solucionado cargando `colortbl` a
secas (da `\rowcolor` singular, que es lo único que se usa) en vez de xcolor con la opción
`table`; se eliminó `\rowcolors` (plural, zebra-striping automático) porque esa sí es exclusiva de
`xcolor[table]` y no tiene sustituto directo sin volver a arrastrar el conflicto.
Resultado final: `poster/sigcite_poster.pdf`, 2 páginas exactas, compila limpio con `tectonic`.

**Actualización (mismo día): el profesor pidió expresamente la versión Word "para poder re-editarla
él mismo"** — así que se volvió a `poster/sigcite_poster_draft.docx`, esta vez llevándole el mismo
pulido visual y el autor único del PDF final (que se había quedado solo en la versión LaTeX): caja
de "Key insight" como tabla de 1x1 celda sombreada (la plantilla ACM no tiene un estilo de callout
nativo), cabeceras de tabla en granate con texto blanco (`shade_cell()`/`set_cell_text_white()`
vía manipulación XML de bajo nivel con `python-docx`, ya que no hay forma de fijar sombreado de
celda desde la API de alto nivel), fila de `boat`/"miss" resaltada en rosa suave, y la gráfica
regenerada con el mismo esquema de dos colores (verde azulado/granate) que la versión LaTeX.
Verificado exportando a PDF vía el mismo flujo de AppleScript+Word ya documentado arriba — sin
bugs nuevos. Los dos entregables (`.docx` y `.pdf`) están ahora sincronizados en contenido y estilo.

**Pendiente**: confirmar con el profesor si la revisión de pósters de SIGCITE es ciego-doble — si
lo es, hay que anonimizar (quitar nombre/afiliación) antes de enviarlo definitivamente.

### El profesor tumba el póster de seguridad: el hallazgo real estaba en la missingness de `boat` (2026-08-13)

Segunda ronda de revisión, mucho más dura: el profesor dijo directamente que el póster "is too
weak to be a poster" y pidió verificar **hoy mismo, antes de nada** si `ρ(boat)=0.420` es un
hallazgo real o un artefacto de implementación, por dos vías posibles: (a) que `boat` estuviera
tomando la rama de Pearson en vez de Cramér's V por algún problema de encoding, y (b) que la
missingez de `boat` (el ~63% de valores ausentes) fuera en sí misma la fuga, y que el pipeline la
estuviera destruyendo antes de calcular la correlación. Advirtió explícitamente: si (b) resulta
cierto, el argumento central del póster **se invierte** — un check de correlación bien
implementado sí atraparía `boat`, y es nuestro propio preprocesado el que lo esconde.

**Verificado con código real, no con argumentos — y (b) resultó ser exactamente cierto:**
- (a) descartado: `boat` es `dtype=object` (1309 filas, 823 nulos = 62.9%, 27 valores no nulos),
  así que `_feature_target_association()` sí toma la rama de Cramér's V, no Pearson. El encoding
  no es el problema.
- (b) confirmado, y es un hallazgo real: `_cramers_v(boat.notna(), survived)` sobre las **1309
  filas completas** da **V=0.948** (98.1% de supervivencia cuando hay número de bote asignado,
  2.8% cuando no) — prácticamente determinista. El `ρ=0.420` que reporta el pipeline hoy es
  exactamente `_cramers_v(boat, survived)` calculado **solo sobre las 486 filas no nulas**,
  porque `_feature_target_association()` hace `.dropna()` antes de calcular la asociación,
  destruyendo la señal más fuerte de la columna. Verificado que el 0.420 reproducido a mano sobre
  el subconjunto no nulo coincide exactamente con lo que ya devolvía el pipeline.
- Mecanismo explicado con precisión (verificado con `pd.factorize` directamente): la razón de que
  `Ĩ(boat)=1.000` y `π(boat)=0.808` ya estuvieran altos en el pipeline actual **no es una ventaja
  de diseño de las señales multi-signal** como se decía hasta ahora en la tesis/paper — es una
  **asimetría accidental**: `compute_leakage_risk_score()` factoriza las categóricas con
  `pd.factorize()` antes de MI/π, y `factorize` codifica los NaN como `-1` (un código propio, no
  como `np.nan`), así que `SimpleImputer(strategy="mean")` no los toca — el missingness sobrevive
  intacto para MI/π. Pero `_feature_target_association()` (la que calcula `ρ`) hace `.dropna()`
  directamente sobre la serie original, sin pasar por `factorize`, así que ahí sí se pierde. Dos
  rutas de código con tratamiento de NaN inconsistente entre sí — ese es el verdadero motivo del
  patrón "MI/π saturan, ρ no", no una propiedad general de la información mutua.

**Bergsma (2013) implementado también, con resultado limpio**: `_cramers_v_bias_corrected()`
añadida a `src/leakage_checks.py` (fórmula exacta que dio el profesor, verificada contra
[el paper original](https://link.springer.com/article/10.1016/j.jkss.2012.10.002) por búsqueda
web) + 4 tests nuevos en `tests/test_leakage_checks.py` (343→347 tests, todos pasan). Resultado
en Titanic: `name` V=0.999→corregido=**0.000** (colapso completo, exactamente lo que predijo el
profesor); `sex`/`embarked` (columnas de baja cardinalidad, "bien comportadas") apenas se mueven
(0.529→0.528, 0.184→0.180), confirmando que la corrección ataca específicamente la dispersión, no
la asociación genuina. **Decisión de alcance**: la función se añadió, se testeó y se documentó,
pero **no** se cambió el comportamiento por defecto de `check_target_leakage()` ni de
`compute_leakage_risk_score()` para usarla automáticamente — hacerlo cambiaría números ya
publicados en `tfm.tex`/`paper.tex` (p.ej. "`name` Cramér's V=0.999" aparece en varias tablas) y
es una decisión mayor que no se tomó unilateralmente; queda pendiente de que el usuario la pida
explícitamente.

**Póster reescrito de cabo a rabo** (`poster/sigcite_poster.tex` y `.docx`), estructura nueva
propuesta por el profesor: título sin ángulo de seguridad ("Cramér's V Fails in Both Directions:
A Diagnostic Case Study of Categorical Leakage Screening on the Titanic Dataset"); Motivación
recortada a 1 párrafo con una sola frase de contexto de seguridad (antes había 3-4 metáforas
repetidas sin threat model, exactamente la crítica del profesor) y desambiguación explícita de
"leakage" = target leakage, no brecha de seguridad; sección nueva "The Two Failures" como núcleo
(Tabla 1 revisada: cardinalidad, singletons, V crudo/corregido, MI, π, L y verdad de terreno para
`name`/`boat`/`sex`/`embarked` — ya no incluye los proxies sintéticos, que el profesor señaló como
"trivially detectable... carry no argumentative weight"; Tabla 2 nueva de missingness); **la
Tabla 2 antigua (ablation de pesos) y la Figura 1 (gráfica del ablation) se eliminaron
completamente** — el profesor demostró que esa tabla era matemáticamente determinista a partir de
la Tabla 1 y no aportaba información nueva; sección "Why Multiple Signals Help" reducida a un
párrafo sin tabla, dejando explícito que "we do not present this weighted sum itself as a
research contribution". Arregladas también las dos citas: Kraskov (nunca citada en el texto, en
la versión anterior) ahora se cita al introducir la estimación de MI; Bergsma añadida como
referencia nueva.

Recompilado y verificado visualmente en LaTeX (`tectonic`, 2 páginas, 0 errores) y en Word (mismo
flujo AppleScript+Word de antes, 3 páginas a una columna ≈ menos de 2 en el formato final a dos
columnas). Ambos formatos sincronizados en contenido.

**Nota para el futuro**: la explicación de la asimetría ρ-vs-(MI,π) descrita arriba es más precisa
y más honesta que la que aparece hoy en `tfm.tex`/`paper.tex` (que la presenta como una ventaja de
diseño del score unificado, sin mencionar que es en parte un accidente de cómo `pd.factorize`
trata los NaN). No se ha tocado la tesis/paper todavía — el usuario no lo ha pedido para esta
ronda, centrada en el póster — pero si se retoma la Opción B (journal) o una futura revisión de la
tesis, esta es una corrección pendiente real, no solo una mejora de redacción.

### Segunda vuelta del profesor: "esto sigue siendo débil" — arreglar el bug de verdad, replicar en NSL-KDD (2026-08-13, misma tarde)

El profesor validó la verificación anterior ("this is exactly the right way to handle it... the
dropna-vs-factorize finding is a better contribution than the thing it replaced") pero encontró
que el póster reformulado **seguía sobrevendiendo el hallazgo**: solo una de las dos fallas
(`name`) es una propiedad genuina de Cramér's V; la otra (`boat`) es un bug de nuestro propio
preprocesado, no una propiedad de la estadística — y publicarlo como "V falla en ambas
direcciones" invita exactamente esa objeción. Pidió 6 cosas: reencuadre honesto + comprobar si
`ydata-profiling`/Deepchecks tienen el mismo bug (30 min), arreglar el bug de verdad (no solo
diagnosticarlo) y reportar antes/después, arreglar la Tabla 1 (L se computaba con la V cruda pese
a recomendar la corregida), dejar explícito que son dos fallos con dos remedios distintos (la
corrección de Bergsma no arregla `boat`), añadir contenido de encaje temático con SIGCITE
(Cybersecurity/IT Ed.) incluyendo una réplica en NSL-KDD, y confirmó que seré primer autor con él
como segundo si esto llega a enviarse.

**Comprobación del ecosistema (los 30 min que pidió) — confirmado, no es solo nuestro bug:**
- `pandas.crosstab` descarta filas con NaN por defecto, verificado empíricamente
  (`pd.crosstab(['a','b',nan,...], ...)` — la fila con NaN desaparece de la tabla de contingencia).
- **`ydata-profiling` tiene exactamente el mismo bug**, confirmado leyendo su código fuente real
  (`ydata_profiling/model/pandas/correlations_pandas.py::_pairwise_cramers()`): llama
  `pd.crosstab(col_1, col_2)` directamente, sin tratar los missing aparte.
- **Deepchecks también**, confirmado empíricamente en el entorno `deepchecks_compat`: PPS de
  `FeatureLabelCorrelation` sobre `boat` con los NaN reales = **0.0006** (prácticamente cero);
  sobre `boat.notna()` como indicador explícito = **0.946**. Mismo fallo, motor distinto (PPS vía
  árboles de decisión, no Cramér's V).
- Esto cambia el argumento de "nuestro pipeline tenía un bug" a "es el comportamiento por defecto
  compartido por el ecosistema de herramientas" — justo lo que pedía el profesor para poder
  reencuadrar el título sin sobrevender.

**El bug arreglado de verdad en `src/leakage_checks.py::_feature_target_association()`** (no solo
diagnosticado): para columnas categóricas, ya no se hace `.dropna()` sobre la fila completa antes
de calcular Cramér's V — solo se descartan filas con **target** desconocido (no se puede asociar
con una etiqueta que no existe), y los valores de **feature** ausentes se codifican como su propia
categoría explícita `"__missing__"`, exactamente igual que ya hacía `pd.factorize()` en la ruta de
MI/π (que asigna a los NaN su propio código `-1`, no `np.nan`). 2 tests nuevos en
`tests/test_leakage_checks.py` (349 tests totales, todos pasan; verificado que ningún test
existente se rompe con el cambio de comportamiento).

**Resultado del arreglo, antes → después (todo verificado con el pipeline real, no a mano):**
- `ρ(boat)`: 0.420 → **0.951**
- `L(boat)`: 0.739 (warning) → **0.925** (**error** — cruza el umbral 0.9)
- `name`/`sex`/`embarked`: sin cambios (no tienen missingness informativa que rescatar)
- Con la V corregida (Bergsma) alimentando L en vez de la cruda: `L(name)`=0.058 (no 0.407),
  separando limpiamente `name` (0.058) de `sex` (0.405) — arreglado el problema de la Tabla 1 que
  señaló el profesor (antes `name` y `sex` puntuaban casi igual bajo L, invitando la objeción obvia
  de que el score no distinguía entre ambos).
- Confirmado explícitamente que la corrección de Bergsma **no** arregla `boat` (0.420→0.351 antes
  del fix; 0.951→0.940 después) — son dos remedios distintos para dos fallos distintos, tal y como
  pidió el profesor que se dejara explícito.
- `Ṽ(name)=0.000` documentado como el suelo recortado del estimador (`max(0, ·)`), no un cero
  exacto casual.

**Réplica en NSL-KDD** (descargado de `jmnwong/NSL-KDD-Dataset` en GitHub, 125.973 filas,
`protocol_type`/`service`/`flag` con cardinalidad 3/70/11 contra una etiqueta binaria
ataque/normal): resultado **negativo**, tal y como el profesor dijo que sería aceptable — V cruda
y corregida son numéricamente indistinguibles en las tres columnas (0.282/0.282, 0.860/0.860,
0.775/0.775), porque ninguna es lo bastante dispersa (solo 1 categoría singleton en `service` de
70, frente a 1.305 de 1.307 en `name`) — confirma que el mecanismo de sesgo depende de la escasez
de observaciones por categoría, no de la cardinalidad por sí sola. 0% de missingness en las tres
columnas (dataset sintético, sin datos ausentes reales), así que esta réplica aísla el mecanismo
de sesgo sin mezclar con el de missingness. Las tres columnas están genuinamente asociadas con el
ataque por diseño (no es leakage), así que no hay caso falso-positivo/negativo que reportar aquí —
solo la confirmación honesta de que el mecanismo no aparece a esta escala.

**Póster reescrito otra vez** (`poster/sigcite_poster.tex`): título nuevo sin la palabra
"security" como eje ("Two Failure Modes in Categorical Target-Leakage Screening: Sparsity Bias
and Complete-Case Deletion"); Tabla 1 ampliada a `table*` (ancho completo, 2 columnas de LaTeX)
con L(V) y L(Ṽ) lado a lado; nueva sección "Negative Replication on NSL-KDD" con su propia tabla;
sección 3 renombrada explícitamente a "Why Multiple Signals Help—And Why One Fix Does Not Cover
Both Failures"; añadidas las dos frases de cierre sobre Titanic en instrucción introductoria de
ML; referencia nueva a Tavallaee et al. 2009 (paper original de NSL-KDD, verificada por búsqueda
web). Compila limpio en `tectonic`, exactamente 2 páginas, verificado visualmente.

**Decisión de alcance explícita**: el profesor dijo "Do not worry about the format, I will
eventually re-edit it on Latex" — así que **no** se actualizó `poster/sigcite_poster_draft.docx`
en esta ronda; ha quedado desactualizado/obsoleto a partir de aquí. El `.tex`/`.pdf` es ahora la
única versión autoritativa. Tampoco se ha tocado `tfm.tex`/`paper.tex` con los números
post-arreglo (`name` V=0.999, `boat` L=0.739) — el profesor fue explícito en que la corrección de
Bergsma no debe alterar retroactivamente las cifras ya publicadas en la tesis, y que la corrección
del framing de complementariedad "genuina" en ambos documentos "is not urgent this week", queda
en la lista para antes de la submission a journal.

**Nota de autoría confirmada por el profesor**: si esto se llega a enviar, Jaime es primer autor,
Yong Zheng segundo.

### Propagación del arreglo del bug de missingness a `paper.tex` y `tfm.tex` (2026-08-13)

Tras arreglar el bug real en `_feature_target_association()` (ver sección anterior) y
reescribir el póster alrededor del nuevo framing honesto, el usuario pidió corregir también
la tesis y el paper con los mismos hallazgos — hasta este punto ninguno de los dos
documentos reflejaba el arreglo, y ambos seguían citando los números pre-fix (`boat`
ρ=0.420, L=0.739; Titanic 80.6/B, 14/22) como si fueran vigentes.

**Verificación previa (antes de tocar ningún documento):** se re-ejecutó `DatasetChecker`
sobre los 12 datasets de evaluación (9 real-world/LRS-validation + los usados en case
studies) para confirmar el alcance exacto del impacto. Resultado: **solo Titanic cambia**
(80.6/B, 14/22 → **75.4/B, 12/22**; sigue siendo grado B). Los otros 8 datasets reales y los
3 sintéticos de validación LRS son idénticos byte a byte. La tabla de ablation de pesos
también se recalculó completa con el ρ corregido: los 5 esquemas de pesos ahora coinciden en
marcar `boat` (Default=0.925, Equal=0.920, Correlation-heavy=0.927, MI-heavy=0.940,
Performance-heavy=0.892 — todos ≥0.7), frente al resultado pre-fix donde el esquema
correlation-heavy fallaba (0.662, bajo el umbral). Además, `check_target_leakage()` ahora
marca **tanto `name` (V=0.999) como `boat` (V=0.951)** — antes solo marcaba `name`, ya que
0.420 estaba por debajo del umbral 0.95.

**Decisión de framing:** en vez de descartar la tabla de ablation antigua o buscar un
ejemplo ilustrativo distinto, se mantuvo `boat` como caso de estudio pero añadiendo una
columna **"boat (pre-fix)"** en la tabla de ablation que reproduce los números originales
(buggy) junto a los corregidos — preserva el valor pedagógico de la tabla (mostrar qué
pasaba con pesos correlation-heavy) mientras dice honestamente que ese resultado era
evidencia de un bug, no una propiedad del peso elegido. La narrativa de "los tres señales
son genuinamente complementarias" se reescribió a la explicación honesta ya usada en el
póster: MI/π ya estaban saturados por una asimetría de implementación
(`pd.factorize` codifica NaN como código propio, no como `np.nan`, así que
`SimpleImputer` nunca los tocaba; pero `_feature_target_association()` sí hacía
`.dropna()` directo), no por una ventaja de diseño del score combinado. El caso de `name`
(sparsity bias genuino de Cramér's V, arreglable con Bergsma) se mantiene como el ejemplo
honesto de "por qué combinar señales sigue teniendo sentido", separado explícitamente del
caso de `boat`.

**Ubicaciones corregidas en ambos documentos** (números, no solo prosa):
introducción/resumen (ejemplo motivador reescrito para contar la historia de missingness +
ρ=0.951, en vez de "ρ=0.420, correlación lo pierde"); tabla y párrafo de "Weight
sensitivity" (con la nueva columna pre-fix); tabla de validación LRS (`tab:lrs`, fila
`boat`: ρ 0.420→0.951, L 0.739→0.925); tabla de readiness scores y su gráfica de barras
(Titanic 80.6→75.4); tabla y prosa de case studies (`boat` ahora también detectado por
`target_leakage`, no solo por el score unificado); párrafo de Discussion/Discussion of
Results ("genuinely complementary" reescrito a "two genuine failure modes... two remedies");
párrafo de Limitations/Current Limitations (Bergsma ya no es future work, está implementado
en `src/leakage_checks.py`, con nota explícita de que no se ha hecho default todavía);
seguridad relacionada — 3 sitios adicionales en `tfm.tex` donde `boat` se usaba como ejemplo
de "fuga que la correlación no puede detectar" para motivar el módulo semántico (resumen
ES/EN + un párrafo de metodología) se corrigieron sustituyendo `boat` por
`discharge_code`/`future_usage` (los ejemplos ya usados de forma consistente en el resto
del documento), porque tras el fix esa afirmación ya no es cierta para `boat` específicamente.

**Conteo de tests actualizado 343→349** en todas las tablas/menciones de `tfm.tex` (la
tabla de distribución por fichero: `test_leakage_checks.py` 50→56 tests, tras sumar los 4
tests de Bergsma + los 2 de missingness añadidos en esta misma ronda de bugs). `paper.tex`
no menciona el conteo total de tests en ningún punto, así que no necesitó cambios ahí.

**Bibliografía**: se añadió una entrada real de Bergsma (2013) a la bibliografía de ambos
documentos (antes solo se citaba como "Bergsma's (2013)" en prosa, sin `\bibitem`/`\cite`,
en ambos ficheros) y se convirtieron las menciones en prosa a `\cite{bergsma2013bias}` /
`\cite{bergsma2013}`.

**No tocado deliberadamente**: la corrección de Bergsma (V̄) sigue sin ser el default de
`check_target_leakage()`/`compute_leakage_risk_score()` — ambos documentos ahora lo dejan
explícito como decisión de alcance, no como omisión. La tabla de fases históricas de
`tfm.tex` (`tab:phases`, 16 comprobaciones por fase) no se reequilibró fila a fila (los 6
tests nuevos de esta ronda no pertenecen a ninguna fase original de las 17); solo se
actualizó el total final con una nota aclaratoria, siguiendo el mismo precedente que la
inconsistencia aritmética preexistente de esta tabla ya documentada en la ronda de Bedrock
(ver nota de 2026-08-06 sobre esta misma tabla).

Ambos documentos recompilados con `tectonic` (0 errores; solo warnings preexistentes de
over/underfull hbox) y verificados visualmente (`paper.pdf`: 15→17 páginas; `tfm.pdf`:
52→55 páginas) leyendo las páginas exactas de las secciones editadas, no solo grep sobre el
`.tex`. Suite completa (349 tests) re-verificada tras los cambios: sigue en verde.

### Tercera vuelta del profesor sobre el póster: la cadena de evidencia + dos citas rotas (2026-08-13, misma tarde)

El profesor validó la investigación anterior (lectura del código fuente de ydata-profiling en
vez de inferir por comportamiento, réplica independiente del mismo fallo en Deepchecks
PPS=0.0006→0.946, atribución correcta en NSL-KDD a "observaciones por categoría, no
cardinalidad") como justo lo que convierte "nuestro pipeline tenía un bug" en "un default
compartido por el ecosistema de herramientas". Pidió 3 arreglos más al póster
(`poster/sigcite_poster.tex`) antes de la compilación final, que él haría a mano en LaTeX:

1. ✅ **La Tabla 1 no sostenía el argumento del §3** — la tabla solo mostraba `boat` post-fix
   (V=0.951), así que la afirmación de que "Ĩ y π ya estaban saturados mientras ρ no" no tenía
   ninguna fila que la respaldara, solo una nota al pie. Arreglado añadiendo una fila
   `boat (pre-fix)` (V=0.420, Ṽ=0.351, L(V)=0.739, L(Ṽ)=0.715, cardinalidad **27** —no 28,
   verificado con `pandas` directamente en vez de copiar el 28 que sugería el profesor en su
   borrador de tabla, ya que pre-fix se cuenta sobre las 486 filas no nulas, 27 categorías
   únicas, no 28) junto a la fila `boat (post-fix)` ya existente (cardinalidad 28, con la
   categoría de missingness). §3 reescrito para señalar la fila en vez de repetir los números
   en prosa.
2. ✅ **Reconciliar 0.948 vs 0.951** — añadida una cláusula a la caption de la Tabla 2
   explicando que 0.948 es `boat.notna()` (indicador binario) contra survived, mientras que
   0.951 (Tabla 1) mantiene las 27 categorías no-missing distintas más una 28ª categoría de
   missingness — dos formas de medir la misma señal, no una contradicción.
3. ✅ **Justificar "legítimamente asociado" en NSL-KDD** — añadida una frase explícita: a
   diferencia de `boat` (post-hoc, asignado después del resultado), `protocol_type`/`service`/
   `flag` son propiedades observables de la conexión en el momento de la clasificación, no
   valores registrados después de etiquetar el ataque — mismo criterio que separa leakage de
   asociación legítima, aplicado consistentemente a ambos datasets.

**Verificación de las 4 referencias pedida explícitamente por el profesor** (agentes de
búsqueda web independientes contra DBLP/ACM DL/APS/journals.aps.org, no memoria): Kaufman
et al. 2012 tenía un error real de título — decía "Formalization" en `paper.tex` y en el
póster, cuando el título correcto (verificado en ACM DL, DBLP, Semantic Scholar) es
"**Formulation**, detection, and avoidance"; corregido en ambos (`tfm.tex` ya tenía la palabra
correcta). Kraskov et al. 2004 le faltaba el número de artículo (**066138**, verificado en
journals.aps.org) en `paper.tex` y en el póster; corregido en ambos (`tfm.tex` ya lo tenía).
Bergsma 2013 y Tavallaee 2009 (esta última la cita nueva que el profesor pidió verificar
especialmente): ambas confirmadas exactas tal cual estaban (autores, título, venue, año,
DOI/páginas) — el aparente desajuste DOI-vs-año de Bergsma (`10.1016/j.jkss.2012.10.002` con
año de cita 2013) es el patrón normal de Elsevier (DOI con el año de disponibilidad online,
cita con el año de la edición impresa), no un error.

**Bug de renderizado real encontrado durante la verificación visual** (no solo de contenido):
`\texttt{FeatureLabelCorrelation}` (23 caracteres, sin puntos de ruptura) desbordaba la
columna estrecha del póster y se solapaba visualmente con el texto de la columna vecina —
arreglado con guiones discrecionales manuales (`\texttt{Feature\-Label\-Correlation}`), que
funcionan en cualquier fuente incluida `\texttt` (a diferencia de la guionización automática,
que no aplica a fuentes de ancho fijo).

**El póster creció a 3 páginas** tras añadir la fila nueva de Tabla 1 y las clarificaciones de
caption — el límite de 2 páginas de SIGCITE es real y no negociable. Recuperado el espacio
sin quitar ningún contenido sustantivo pedido por el profesor: se podó prosa redundante que
las propias tablas ya hacían innecesaria (p.ej. la restated "antes→después" de `boat` después
de la Tabla 2, ya visible en las dos filas de la Tabla 1; la frase de `sex`/`embarked` "barely
move" ya visible numéricamente en la tabla), se comprimió el párrafo de "Practical
Implications", y se ajustó el espaciado tipográfico (`\intextsep`, `\abovecaptionskip`,
padding del cuadro `mdframed` de "Key insight", tamaño de fuente de la bibliografía a
`\small`) — todo dentro de los mismos márgenes de página de `acmart`/`sigconf`, sin tocar la
geometría. Verificado visualmente página por página tras cada recorte (no solo conteo de
páginas) hasta confirmar 2 páginas exactas sin solapamientos ni desbordes.

**Pendiente explícito**: el profesor dijo que hará él mismo la compilación final en LaTeX, así
que no se ha tocado `poster/sigcite_poster_draft.docx` (ya desactualizado desde la ronda
anterior) ni se ha enviado nada más allá del PDF recompilado.

---

## Tesis TFM (tfm.tex)

- **Archivo:** `tfm.tex` / `tfm.pdf` (50 páginas, formato UPM — Máster Universitario en Ingeniería de Telecomunicación, ETSIT).
- **Compilar:** `tectonic tfm.tex`.
- **Estilo visual (actualizado 2026-06-29 para matchear `tfm-upm.pdf`/TFG de referencia exactamente):**
  - Capítulos/Anexos: barra naranja (`upmOrange`) vía macro `\upmchapter{}`, título en **MAYÚSCULAS**. La barra usa `\colorbox` envolviendo un `\parbox` (macro `\upmchapterbar`) que se ajusta automáticamente a 1 o 2 líneas — **no usar una altura fija**, rompía visualmente con títulos largos (ej. el Anexo A en 2 líneas).
  - Secciones: naranja, MAYÚSCULAS, con regla inferior (`\titlerule`).
  - Subsecciones: azul (`upmBlue`), **MAYÚSCULAS, con regla inferior fina** (antes no tenían regla ni mayúsculas — este es el detalle que más se parecía al "A.2.1 IMPACTOS ÉTICOS" de la plantilla oficial).
  - Subsubsecciones: cursiva azul, mayúsculas.
  - Índice (ToC) y cabeceras de página: se mantienen en Title Case normal (solo el título en la propia página va en mayúsculas) para legibilidad.
  - `tfm-upm.pdf` sigue siendo el fichero de referencia oficial, **no tocar**, sin trackear en git.
- **Estructura actual (5 capítulos + 2 anexos — verificada por grep de `\upmchapter{}` el 2026-08-06,
  ver nota de corrección más abajo):**
  1. Introduction and Objectives
  2. State of the Art and Related Work
  3. Development (Arquitectura, Metodología, Implementación — incl. §3.3 "Tools Used During
     Development" y §3.3(bis) "Tools Used During Testing and Evaluation" como **subsecciones**,
     no como capítulo propio, ver nota abajo — , Resumen de fases)
  4. Results
  5. Conclusions and Future Research (incluye Bibliografía como capítulo automático vía `thebibliography`)
  Anexo A: Ethical, Economic, Social, and Environmental Aspects
  Anexo B: Economic Budget (tabla única: Cost of Labor / Cost of Material Resources / General Overheads + Industrial Profit / Subtotal + VAT / Total — ver detalle abajo)
  - Front matter: Resumen, Summary, **Acronyms**, Contents, List of Figures, List of Tables.
  - ⚠️ **Nota (2026-08-06):** esta entrada decía antes "6 capítulos" con un capítulo 4 "Tools"
    separado (creado tras el feedback del ponente del 2026-06-24, ver sección de abajo). En algún
    punto posterior no documentado, "State of the Art" pasó a ser capítulo propio (2) y "Tools" se
    replegó a subsecciones dentro de "Development" (3) — el `\upmchapter{}` de Tools ya no existe
    en `tfm.tex`. La lista de arriba refleja el `tfm.tex` real tal y como está hoy; si se vuelve a
    reestructurar, actualizar aquí.

### Feedback del Ponente/tutor en España (2026-06-24)

1. ✅ **Figura 1 del paper rota** (cajas/líneas superpuestas) → corregido en `paper.tex` (nodo `outbox` minimum height 0.6→0.9cm + espaciado fila 6 de 0.9→1.3cm) → commit `6eff47b`.
2. ✅ **Estructura de capítulos "todo metido en el 2"** → resuelto usando como referencia el TFG previo del autor (`TFG-JaimeCanoMoraño_vf.pdf`, ETSIT-UPM 2023-24): se extrajo el capítulo 4 "Tools" replicando el patrón 4.1/4.2 del TFG, en vez de partir Development en "state of the art / architecture" (el propio TFG de referencia tampoco separa eso, mantiene un único capítulo "Desarrollo").
3. ✅ **Presupuesto "un poco raro"** → Anexo B reescrito con estructura estándar de presupuesto de ingeniería española (mano de obra + recursos materiales = Costes Directos; +gastos generales; +beneficio industrial; +IVA).
4. ✅ **Página de Acrónimos** añadida al front matter, replicando convención del TFG de referencia.
- Commit de la restructuración de `tfm.tex`: `37e2ef3`.

### Ronda de pulido visual y presupuesto (2026-06-29)

1. ✅ **Figuras del paper/tesis mejoradas** (commits `48d9b4e`, `88a4d9f`): caja "Recommendations (20)" del diagrama de pipeline acortada a "Code-level / Recs. (20)" para no quedar más grande que sus vecinas. En el gráfico de barras de readiness scores: la leyenda se movió fuera del área de datos (ya no tapa la primera columna), y las líneas de umbral de grado A/B/C pasaron de ser puntos repetidos por barra (causando "columnas" falsas, bug real de `ybar`+`addplot` de 2 puntos) a líneas `\draw` continuas con `axis cs:`, coloreadas verde/naranja/rojo (A/B/C).
2. ✅ **Pulido de redundancia narrativa + auditoría final de cifras** (commit `0b0cdd3`): la explicación completa del caso `boat`/`name` (Titanic) se contaba 3 veces por documento; recortada para que el caso de estudio remita a Discussion. Cifras "29 checks" sueltas sin el framing "checklist" corregidas. Añadidos CD/VAT a Acrónimos (CI se añadió y luego se quitó, ver punto 4).
3. ✅ **Estilo de cabeceras de tfm.tex** (commit `b4ea399`) → ver sección de estilo visual arriba.
4. ✅ **Presupuesto bajado y simplificado** (commit `05fdaac`), siguiendo el ejemplo visual de Anexo B del TFG de referencia: tabla única por secciones (en vez de 3 tablas separadas), beneficio industrial calculado sobre CD (no CD+CI, igual que el TFG de referencia), tarifa bajada de €35/h a €15/h (manteniendo las 312h ya justificadas en la Tabla 2.5 de fases), 2 partidas de material (portátil amortizado + tokens API) en vez de 3. **Total: €16,436.92 → €7,086.24.** Acrónimo "CI" eliminado (ya no se usa "Indirect Costs" como concepto propio en la tabla simplificada).

### Auditoría final de consistencia + pulido de redacción en paper.tex y tfm.tex (2026-08-06)

Lectura completa de ambos ficheros (`paper.tex` 1397 líneas, `tfm.tex` 2735 líneas) línea por
línea, cruzando cifras entre ambos documentos y contra el código/tests reales. No solo pulido de
prosa: aparecieron **bugs de contenido reales**, no solo de redacción. Ninguno de estos ficheros
está trackeado en git (ver sección de GitHub cleanup más abajo), así que estos cambios solo
existen en local — no hay commit asociado.

1. ✅ **Caption "two models for Titanic, three for the others"** (Tabla de readiness scores, en
   ambos documentos) — Pima Diabetes también usa 2 modelos (`diabetes_config.yaml` solo tiene
   `logistic_regression`+`random_forest`, igual que `titanic_config.yaml`; los datasets sintéticos
   usan `config.yaml` con los 3). El caption decía "three for the others" implicando que Diabetes
   tenía 3 — la tabla (19/22) siempre estuvo bien, el caption estaba mal. Corregido en los dos
   ficheros para nombrar explícitamente "Titanic and Pima Diabetes" vs "the synthetic datasets".
2. ✅ **Error gramatical en español** (resumen de `tfm.tex`): "Titanic, Adult Census **e** German
   Credit" → "**y** German Credit" ("e" solo sustituye a "y" ante sonido /i/, no ante "German").
3. ✅ **"13 test files" vs 14 reales** (`tfm.tex`, dos menciones) — la propia tabla `tab:tests` ya
   listaba 14 ficheros que suman 343 correctamente; solo el texto decía "13". Corregido a 14.
4. ✅ **Nombre de fichero incorrecto en la tabla de tests**: `test_drift_checks.py` → el fichero
   real es `tests/test_drift.py` (verificado con `ls tests/`).
5. ✅ **Ejemplo de plugin con firma de API incorrecta** (`tfm.tex`, Listing 3.4): usaba
   `@register_check("my_custom_check")` (un argumento posicional); la firma real en
   `src/plugins.py` es `register_check(phase: str, name: str)` — dos kwargs. Corregido el listing
   completo (decorador + los `CheckResult(...)` internos, renombrados a `"no_future_dates"` para
   consistencia).
6. ✅ **Descripción de pestañas de Streamlit duplicada y no sincronizada** (`tfm.tex`, §3.1 "User
   Interfaces" en el capítulo Development) — una versión antigua/distinta de la que ya existía
   correctamente en §3.3.4 "Web Interface" (esta última ya se había corregido en una sesión previa,
   ver más abajo). La de §3.1 aún decía "a eight-tab dashboard: dataset overview, quality checks,
   leakage detection, feature analysis, sufficiency, impact analysis, and a readiness score summary"
   (7 ítems, ninguno coincide del todo con las 8 pestañas reales). Reescrita para coincidir
   exactamente con la lista real: Quality, Leakage, Features, Sufficiency, Drift, Semantic,
   Recommendations, Download.
7. ✅ **Gramática "a eight-tab" → "an eight-tab"** (2 ocurrencias en `tfm.tex`, "eight" empieza por
   sonido vocálico).
8. ✅ **Cifra de runtime fabricada/no verificada**: el Anexo de Environmental Considerations
   afirmaba que Adult Census (48.842 filas) corre en "under ten seconds", citando
   `Table~\ref{tab:runtime}` — esa tabla solo cubre hasta 5.300 filas, nunca midió Adult Census.
   Medido en vivo (`DatasetChecker.run()` sobre `data/raw/adult.csv`, config por defecto, 3 modelos
   de impact analysis): **51.77 s**, reproducible entre ejecuciones (semillas fijas). Corregido el
   texto para citar la tabla solo donde aplica (hasta 5.300 filas) y dar la cifra real medida
   ("under a minute") para Adult Census, sin atribuirla a una tabla que no la contiene.
9. ✅ **Partida de presupuesto con proveedor LLM incorrecto** (Anexo B, `tfm.tex`): la tabla listaba
   "Azure OpenAI API tokens" como coste de material, pero Azure **nunca se usó** (nunca se
   obtuvieron credenciales, ver sección de `semantic_leakage.py` más arriba) — el proveedor real
   con coste incurrido es AWS Bedrock. Corregido a "AWS Bedrock API tokens".
10. ✅ Verificados contra el código real y confirmados correctos (sin cambios): el listing del SDK
    (`checker.set(leakage_checks__target_leakage__correlation_threshold=...)`,
    `checker.top_recommendations(priority=...)`) coincide exactamente con `src/checker.py`; el
    excerpt de `config.yaml` coincide con el fichero real; el conteo "14 source modules"/"21 checks
    total" en ambas tablas de módulos ya era correcto.

Ambos documentos recompilados con `tectonic` (0 errores en los dos) y cada corrección verificada
visualmente leyendo la página exacta del PDF resultante (no solo `pdftotext`).

**Nota pendiente:** la sección "Estructura actual" de este mismo fichero (justo arriba) tenía una
descripción de la estructura de capítulos de `tfm.tex` desactualizada (mencionaba un capítulo
"Tools" que ya no existe como tal) — corregida en el mismo pase, ver la nota inline de arriba.

### Pasada de control visual sobre tfm.pdf (2026-08-06, mismo día)

Tras la auditoría de contenido de arriba, pasada adicional centrada en el **PDF compilado en sí**
(no solo el `.tex`): revisadas visualmente portada, página de tribunal, portada interior, Resumen,
Summary, Acronyms, Índice, Lista de Figuras/Tablas, diagrama de pipeline (Fig. 3.1), gráficas de
barras (Fig. 4.1/4.2) y listings de código — página por página con la tool `Read` sobre el PDF, no
solo `pdftotext`.

1. ✅ **Bug real encontrado**: warning `Package fancyhdr Warning: \headheight is too small (12.0pt)`
   repetido 50 veces en el log (una por página) — la cabecera con el logo (`figures/upm_logo.png`,
   9pt de alto) necesita 14.5pt y `geometry` solo reservaba 12pt. Corregido añadiendo
   `headheight=14.5pt` a las opciones de `\usepackage[...]{geometry}` (línea ~6 de `tfm.tex`).
   0 warnings de este tipo tras el fix; verificado que la cabecera se ve igual visualmente (no hubo
   regresión al reservar 2.5pt más).
2. ❌ **Falsa alarma descartada**: el logo ETSIT/UPM de la portada naranja (página 1) se ve con
   texto diminuto solapado entre "ETSIT" y "UPM" — parecía un bug de la imagen `upm_cover_bg.png`.
   Comparado pixel a pixel contra `tfm-upm.pdf` (el PDF de referencia oficial, que no se toca): es
   **idéntico** — así es como se ve el crest institucional real de la ETSIT-UPM en el original. No
   tocar.
3. Resto de páginas revisadas (front matter completo, capítulos 1-5, anexos A y B, figuras y
   tablas): sin defectos visuales — sin overlaps, sin referencias rotas ("??"), sin figuras
   cortadas. El espacio en blanco al final de algunas páginas antes de un `\upmchapter{}` es
   comportamiento normal de LaTeX (los capítulos siempre empiezan en página nueva), no un bug.

### Dos bugs de maquetación reales encontrados por el usuario y corregidos (2026-08-06)

Tras la pasada de control visual de arriba, el usuario detectó dos problemas reales que se me
habían pasado:

1. ✅ **RESUMEN/SUMMARY desperdiciaban una página cada uno** — el resumen en español ocupaba la
   p.3 completa + "Palabras clave" solo en la p.4 (con el resto en blanco); igual con
   Summary/Keywords. Causa: `\titlespacing*{\chapter}` tenía demasiado espacio antes/después
   (`{0pt}{6pt}{14pt}`) y `headsep` (paquete `geometry`) usaba el default de 25pt. Corregido a
   `\titlespacing*{\chapter}{0pt}{0pt}{8pt}` + `headsep=16pt` en las opciones de `geometry` —
   ambos resúmenes ahora caben en una sola página cada uno (con "Palabras clave"/"Keywords"
   incluido). Cambio global (afecta a todos los `\chapter`/`\chapter*`), verificado que no rompe
   ningún otro chapter opener del documento. **Documento pasó de 53 a 52 páginas.**
2. ✅ **Listing 3.1 (Python SDK) partido por la Figura 3.1 (pipeline)** — la figura, al no caber
   en el punto exacto donde se declaraba en el `.tex`, se pospuso (comportamiento normal de los
   floats de LaTeX) y acabó renderizándose *en medio* del listing de código que venía después en
   el texto (líneas 1-7 del listing en una página, la figura entera insertada a continuación, y
   las líneas 8-10 del listing debajo de la figura) — visualmente parecía una superposición.
   Corregido añadiendo `\usepackage{placeins}` + `\FloatBarrier` justo después de la figura del
   pipeline (fin de la subsección "Pipeline Architecture"): esto obliga a LaTeX a colocar la
   figura *antes* de continuar con el texto siguiente, en vez de dejarla flotar más allá. Verificado
   que la Figura 3.1 y el Listing 3.1 completo (10 líneas) ahora caen juntos y en orden en la misma
   página, sin interrupciones.

Ambos verificados visualmente en el PDF recompilado tras el fix (no solo por conteo de páginas).
Si en el futuro aparece un patrón similar (figura flotante que se cuela entre listing y su
continuación), el mismo `\FloatBarrier` es la solución estándar — considerar añadirlo también tras
otras figuras del documento si se repite el problema.

3. ✅ **"Contents" compartía página con "Acronyms"** — a diferencia de `\chapter*{RESUMEN}` /
   `\chapter*{SUMMARY}` / `\chapter*{ACRONYMS}` (que sí fuerzan salto de página por sí solos vía el
   mecanismo interno de `\chapter*`), el `\tableofcontents` estándar de la clase `report` —pese a
   llamar también a `\chapter*{\contentsname}` internamente— no estaba forzando el salto de página
   en este documento (motivo exacto no aislado del todo: posible interacción entre `tocloft` y el
   `\chapter*` interno de `\tableofcontents`), y además "Contents" se renderiza con el estilo
   `\chapter*` plano de la clase, no con la barra naranja custom — esto último no se tocó, el
   usuario solo pidió el salto de página. Arreglado con un `\clearpage` explícito justo antes de
   `\tableofcontents` (línea ~381). Verificado: "Contents" ahora arranca en página propia, List of
   Figures/Tables siguen con su barra naranja normal, conteo total de páginas sin cambios (52).

---

## Presentación de defensa (25 min, en inglés) — 2026-07-09 / 2026-07-29

Tres artefactos, todos en la raíz del repo y en GitHub:

| Artefacto | Fuente | Cómo regenerar | Último commit relevante |
|-----------|--------|----------------|--------|
| `presentation.pdf` (Beamer, 23 slides, 16:9 — **desactualizado**, ver nota abajo) | `presentation.tex` | `tectonic presentation.tex` | `a4ceca1` |
| `presentation.pptx` (**versión principal**, 30 slides, editable) | `scripts/build_presentation_pptx.py` (python-pptx) | `python scripts/build_presentation_pptx.py` | `cd4bba9` |
| `presentation_script.pdf` (guion del presentador, 9 págs.) | `presentation_script.tex` | `tectonic presentation_script.tex` | `cd4bba9` |

⚠️ **`presentation.pdf` (versión Beamer) quedó desactualizada** tras las rondas de edición del
2026-07-29 (slide de contexto ML + resultados reales de Bedrock) — solo se mantuvo el PPTX y su
guion. Si se necesita, regenerar `presentation.tex` a mano con el mismo contenido antes de compilar.
El propio fichero `presentation.pdf` además aparece borrado del working tree (pendiente de decisión
del usuario, no relacionado con el contenido).

### Estructura (compartida por PPTX y guion, ~25:15 total)

1. **Motivation & Objectives** (~7:00) — agenda, **slide de contexto "Why Machine Learning Matters Today"** (KPIs de adopción/mercado/coste de mala calidad de datos + gráfica de adopción 2022-2025, fuentes McKinsey/Statista/Gartner), problema/definición leakage, ejemplo 1.000→0.952, gap de herramientas, 4 contribuciones + KPIs.
2. **System & Methodology** (~7:40) — arquitectura, 3 interfaces + SDK, catálogo 20 checks, **LRS Eq. + ablation**, LLM semántico + **slide de resultados reales de evaluación en vivo** (ya no hay caveat de mock), readiness score.
3. **Experimental Results** (~3:40) — 12 datasets, gráfica readiness (teal=A, naranja=B), caso Titanic (`name` V=0.999 error, `boat` L=0.74 warning), impact analysis.
4. **Benchmark** (~1:50) — cobertura 29/29 vs 8–11 (con nota de fairness), tabla detección 4/4 vs 0–1/4.
5. **Demo & Conclusions** (~5:05) — demo Streamlit en vivo con titanic.csv (3 min, plan B: grabación/HTML), conclusiones, future work (ya no incluye "live LLM evaluation", sustituido por "larger multi-provider benchmark"), cierre.

### Slide de contexto "Why Machine Learning Matters Today" (añadida 2026-07-29)

Slide 4 del deck, con estadísticas reales verificadas por búsqueda web (no inventadas, dado que
se presentan ante un tribunal académico): adopción de IA **88% en 2025** vs. 50% en 2022
(McKinsey State of AI survey), mercado global de IA **$260B → $1.2T+** (2025→2030, Statista
Market Insights), coste medio de mala calidad de datos **$12.9M/año/organización** (Gartner).
Gráfica nativa de barras (2022-2025, última barra en naranja) + nota de fuentes al pie. El
tiempo se compensó recortando la narración (no el contenido visual) de 5 slides que lo permitían
sin perder valor: Gap in Tools, Objectives, Architecture, Check Catalog, Experimental Setup.

### Módulo semántico: de "honest caveat" a resultado real (2026-07-29)

La slide que antes se llamaba "Semantic Module: Evaluation and an Honest Caveat" (bloque rojo de
transparencia) pasó a ser **"Semantic Module: Live Evaluation Results"**: dos tablas (P/R/F1 por
umbral + las 3 discrepancias reales del modelo) en vez del caveat, sin bloque rojo. Ver la sección
de integración de AWS Bedrock más arriba para los números y el porqué. El anexo de Q&A del guion
también se actualizó: la pregunta "¿La evaluación del módulo LLM es real?" pasó de "No" a "Sí".

### PPTX — sistema de diseño (`scripts/build_presentation_pptx.py`)

- Identidad UPM: `upmBlue #243F60` (estructura), `upmOrange #FF8000` (acentos), teal para resultados positivos.
- Componentes reutilizables en el script: `chrome()` (cabecera con kicker de sección + pie con nº de slide en chip), `divider()` (separador full-bleed con número gigante y **5 puntos de progreso**), `kpi_card()`, `numbered_card()`, `styled_table()` (cabecera azul + zebra), `code_box()`, `box_node()`+`arrow()` (diagrama de arquitectura con conectores flecha vía XML `a:tailEnd`).
- Portada y cierre full-bleed azul; agenda con tiempos; gráficas **nativas** de PowerPoint (editables), barras coloreadas por punto (`series.points[i].format.fill`).
- **Gotchas de python-pptx aprendidos:** el texto de autoshapes se centra por defecto → fijar `par.alignment` SIEMPRE; charts con varios `addplot`/series y `symbolic coords` necesitan `bar shift=0pt` en pgfplots o eje con `minimum_scale/major_unit` explícitos en pptx; `shadow.inherit = False` en toda forma; `number_format="0.0"` en data labels NO cambia el separador decimal (lo impone el locale del PowerPoint del usuario → "98,8" con locale español, aceptable); para gráficas de porcentaje usar `number_format='0"%"'` (formato Excel con literal entre comillas) en vez de multiplicar por 100.
- **Workflow de verificación visual sin abrir PowerPoint a mano:** AppleScript `save pres in POSIX file ... as save as PDF` (PowerPoint instalado en el Mac del usuario) → leer el PDF con la tool Read. Ojo: la primera vez el `open` puede dar timeout de AppleEvent (-1712) pero el export termina igualmente; cerrar presentaciones abiertas antes de regenerar (`close every presentation saving no`).

### Guion (`presentation_script.pdf`)

- Texto hablado **en inglés** (~110–115 wpm), acotaciones escénicas en español en cursiva, chips de tiempo por slide + acumulado.
- Página 1: chuleta de números clave, incluyendo el bloque nuevo de contexto ML (88%/2022→2025, $260B→$1.2T+, $12.9M) y el bloque de evaluación semántica en vivo (medium 1.00/1.00/1.00, high 0.917/0.846/0.880).
- Puntos de control: a mitad (slide 17) deberías ir por ~14–15 min; si no, comprimir slides 11 (Interfaces) y 24 (Feature Coverage) a una frase, o recortar la demo a 2 min.
- Frase de respaldo si falla la demo: *"In the interest of time, let me show you a pre-recorded run."*
- Anexo: 10 preguntas probables del tribunal con respuesta preparada (pesos LRS→ablation, evaluación **real** del LLM vía Bedrock, umbrales, escalabilidad Adult 48k, por qué GPT-4o-mini/Claude y no un modelo local, fairness del benchmark, falsos positivos vs features predictivas, aclaración 20/21/23/29, bug Heart Disease, por qué citar cifras de mercado en una tesis técnica).
- Gotcha LaTeX: con `helvet`+T1 el carácter `·` literal se renderiza mal ("ů") → usar `$\cdot$`.
- Las 3 frases tachadas a mano por el usuario en una revisión del PDF se eliminaron del `.tex` (mención IIT/ETSIT en la portada, "twenty-five" en la agenda, "reproducibility" en la slide del problema) — si se vuelve a anotar el PDF a mano, avisar para repetir el proceso.

### Pendiente (si el usuario lo pide)

- Grabar el vídeo/capturas de respaldo de la demo de Streamlit.
- Cambiar el separador decimal de las data labels si se presenta con PowerPoint en locale inglés.
- Regenerar `presentation.pdf` (Beamer) si se quiere mantener sincronizado con el PPTX, o decidir retirarlo.
