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
| **LLM quantitative evaluation** | `data/semantic_benchmark.json`, `scripts/evaluate_semantic_leakage.py` | §4.7 Table 8 (P=1.00, R=0.93, F1=0.96) |
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

# Evaluación cuantitativa del módulo semántico (mock, no requiere API key)
python scripts/evaluate_semantic_leakage.py --mock

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
- **Estructura actual (6 capítulos + 2 anexos):**
  1. Introduction and Objectives
  2. Development (Estado del arte, Arquitectura, Metodología, Implementación, Resumen de fases)
  3. Results
  4. **Tools** (separado de Development: 4.1 Tools Used During Development, 4.2 Tools Used During Testing and Evaluation)
  5. Conclusions and Future Research (incluye Bibliografía como capítulo automático vía `thebibliography`)
  Anexo A: Ethical, Economic, Social, and Environmental Aspects
  Anexo B: Economic Budget (tabla única: Cost of Labor / Cost of Material Resources / General Overheads + Industrial Profit / Subtotal + VAT / Total — ver detalle abajo)
  - Front matter: Resumen, Summary, **Acronyms**, Contents, List of Figures, List of Tables.

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
