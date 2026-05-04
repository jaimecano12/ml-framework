# CLAUDE.md — ml-framework (TFM)

## Proyecto

**"An Automated Framework for Dataset Quality Assessment and Data Leakage Detection in Machine Learning"**
TFM de Jaime Cano — Python 3.13, entorno miniconda (`/opt/miniconda3`).
Repositorio: https://github.com/jaimecano12/ml-framework

---

## Reglas de trabajo

- **Siempre** hacer `git commit + git push` al terminar cada fase o bloque de trabajo.
- Usar `python -m pip install` y `python -m pytest` (nunca `pip` ni `pytest` a secas: apuntan a Python 3.12, no al entorno activo).
- Un test roto se arregla antes de seguir; nunca se omite con `--ignore` o `xfail` sin justificación.

---

## Estado actual — COMPLETO

| Fase | Módulo | Tests | Estado |
|------|--------|-------|--------|
| 1 — Scaffold | `src/utils.py`, `main.py`, `requirements.txt` | 16 | ✅ |
| 2 — Config YAML | `src/config.py`, `configs/config.yaml` | 26 | ✅ |
| 3 — Quality checks | `src/quality_checks.py` | 44 | ✅ |
| 4 — Leakage detection | `src/leakage_checks.py` | 41 | ✅ |
| 5 — Impact analysis | `src/impact_analysis.py` | 21 | ✅ |
| 6 — Report generation | `src/reporting.py`, `src/templates/report.html.j2` | 14 | ✅ |
| 7 — Experiments | `scripts/generate_data.py`, `scripts/run_pipeline.py` | — | ✅ |

**Total: 178 tests, 178 passed.**

---

## Arquitectura

```
ml-framework/
├── configs/config.yaml          — YAML config (dataset, checks, impact, reporting)
├── data/
│   ├── raw/                     — clean_dataset.csv, dirty_dataset.csv, leaky_dataset.csv
│   ├── processed/
│   └── external/
├── reports/                     — HTML reports generados automáticamente
├── scripts/
│   ├── generate_data.py         — genera los 3 datasets sintéticos
│   └── run_pipeline.py          — demo end-to-end sobre los 3 datasets
├── src/
│   ├── utils.py                 — logger, load_dataset(), CheckResult, FrameworkReport
│   ├── config.py                — load_config(), _deep_merge(), _validate()
│   ├── quality_checks.py        — 6 checks + run_all_quality_checks()
│   ├── leakage_checks.py        — 4 checks + run_all_leakage_checks()
│   ├── impact_analysis.py       — run_impact_analysis() (baseline vs cleaned CV)
│   ├── reporting.py             — generate_report() → HTML con Jinja2 + matplotlib
│   └── templates/report.html.j2 — plantilla HTML del informe
├── tests/
│   ├── test_utils.py
│   ├── test_config.py
│   ├── test_quality_checks.py
│   ├── test_leakage_checks.py
│   ├── test_impact_analysis.py
│   └── test_reporting.py
└── main.py                      — CLI: --config, --dataset, --output-dir, --log-level
```

---

## Módulos clave

### `src/utils.py`
- `setup_logger(log_level, log_file)` — configura loguru
- `load_dataset(path, **kwargs)` → `pd.DataFrame` — CSV / Parquet / Excel
- `CheckResult` (dataclass) — resultado de un check individual:
  - `check_name`, `passed`, `severity` (`info|warning|error`), `message`, `details`, `affected_columns`
- `FrameworkReport` (dataclass) — informe agregado:
  - `quality_results`, `leakage_results`, `impact_results`, `metadata`
  - `.all_results()`, `.failed_checks()`, `.summary()`

### `src/config.py`
- `load_config(path)` — carga YAML, aplica `_deep_merge` con `_DEFAULTS`, valida
- `get_section(config, section)` — devuelve un bloque del config
- `_deep_merge(base, override)` — merge recursivo; las listas se reemplazan (no se fusionan)
- `_DEFAULTS` — fuente única de verdad para valores opcionales

### `src/quality_checks.py`

| Función | Qué detecta |
|---------|-------------|
| `check_missing_values` | Columnas con tasa de NaN > threshold |
| `check_duplicates` | Filas exactamente duplicadas |
| `check_outliers` | Outliers por IQR o z-score en columnas numéricas |
| `check_class_imbalance` | Clase minoritaria < threshold en target |
| `check_constant_features` | Columnas con un único valor |
| `check_low_variance` | CV (std/\|mean\|) < threshold en columnas numéricas |

Todas devuelven `CheckResult`. El orquestador es `run_all_quality_checks(df, target_col, config)`.

**Decisión de diseño:** low_variance usa coeficiente de variación (CV = std/|mean|) en lugar de varianza normalizada min-max, porque min-max siempre mapea al rango [0,1] y no captura columnas casi-constantes correctamente.

### `src/leakage_checks.py`

| Función | Qué detecta |
|---------|-------------|
| `check_target_leakage` | Pearson \|r\| (numérico) / Cramér's V (categórico) ≥ threshold |
| `check_train_test_overlap` | Filas duplicadas que cruzarían train/test en split simulado |
| `check_temporal_leakage` | Dataset desordenado temporalmente (requiere date_column) |
| `check_id_column_leakage` | Columnas string/int con ratio de únicos ≥ threshold |

**Decisión de diseño:** `check_id_column_leakage` omite columnas float — los features continuos tienen alta cardinalidad de forma natural y no son proxies de ID.

Orquestador: `run_all_leakage_checks(df, target_col, config)`.

### `src/impact_analysis.py`
- `run_impact_analysis(df, target_col, report, config)` → `list[CheckResult]`
- Para cada modelo configurado:
  1. **Baseline**: CV sobre el dataset completo
  2. **Cleaned**: CV tras eliminar columnas problemáticas (target_leakage, id_column_leakage, constant_features) y deduplicar
  3. Delta = cleaned − baseline. Si delta < −0.05 → `passed=False` (degradación significativa)
- Helper `_extract_problem_columns(report)` extrae las columnas afectadas del informe previo
- Preprocesamiento: `SimpleImputer → StandardScaler → modelo` (Pipeline de sklearn)
- Soporta: `logistic_regression`, `random_forest`, `xgboost`
- Scorer multi-clase automático: `roc_auc` → `roc_auc_ovr_weighted`, `f1` → `f1_weighted`

### `src/reporting.py`
- `generate_report(report, output_dir, config)` → `Path` del HTML generado
- Plots (base64 PNG embebidos en el HTML):
  - Barras: checks pasados/fallidos por fase
  - Pie: distribución de severidades
  - Barras agrupadas: baseline vs cleaned por modelo
- Template Jinja2: `src/templates/report.html.j2` — HTML autocontenido, CSS inline, sin dependencias externas

---

## Config YAML (`configs/config.yaml`)

```yaml
dataset:
  path: data/raw/dataset.csv
  target_column: target

quality_checks:
  missing_values:   { threshold: 0.05 }
  outliers:         { method: iqr, threshold: 3.0 }
  class_imbalance:  { threshold: 0.1 }
  low_variance:     { threshold: 0.01 }

leakage_checks:
  target_leakage:     { correlation_threshold: 0.95 }
  train_test_overlap: { test_size: 0.2, random_state: 42 }
  temporal_leakage:   { date_column: null }
  id_column_leakage:  { cardinality_threshold: 0.95 }

impact_analysis:
  models: [logistic_regression, random_forest, xgboost]
  cv_folds: 5
  metrics: [accuracy, roc_auc, f1]

reporting:
  output_dir: reports/
  include_plots: true
```

---

## Ejecución

```bash
# Instalar dependencias (en entorno miniconda)
python -m pip install -r requirements.txt

# Generar datos sintéticos
python scripts/generate_data.py

# Demo end-to-end (genera 3 informes HTML en reports/)
python scripts/run_pipeline.py

# Pipeline completo sobre un dataset propio
python main.py --dataset data/raw/mi_dataset.csv --config configs/config.yaml

# Tests
python -m pytest tests/ -v
```

---

## Resultados del experimento (Phase 7)

| Dataset | Checks totales | Pasados | Fallidos | Δ accuracy (LR) |
|---------|---------------|---------|----------|-----------------|
| clean_dataset | 11 | 11 | 0 | +0.000 |
| dirty_dataset | 11 | 6 | 5 | +0.000 |
| leaky_dataset | 11 | 4 | 7 | −0.048 |

- **clean_dataset**: ningún problema detectado, baseline = cleaned.
- **dirty_dataset**: 5 problemas de calidad (missing, outliers, constante, baja varianza, desbalance). Sin leakage.
- **leaky_dataset**: 7 fallos incluyendo `target_leakage` (4 features) y `id_column_leakage` (2 columnas). Al eliminar las features problemáticas, la accuracy baja 4.8 pp — el modelo se apoyaba en leakage.

---

## Dependencias principales

```
pandas, numpy, scipy, scikit-learn, xgboost
pyyaml, jinja2, matplotlib, seaborn
loguru, pytest
```
