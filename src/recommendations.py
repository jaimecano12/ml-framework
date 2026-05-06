"""Recommendations engine — generates actionable suggestions from failed checks (Phase 8)."""

from __future__ import annotations

from .utils import CheckResult, FrameworkReport, Recommendation, logger

# ---------------------------------------------------------------------------
# Per-check recommendation generators
# ---------------------------------------------------------------------------

def _rec_missing_values(r: CheckResult) -> list[Recommendation]:
    flagged = r.details.get("flagged_columns", {})
    if not flagged:
        return [Recommendation(
            check_name="missing_values",
            priority="medium",
            action="Investigate and handle missing values",
            rationale="Missing values detected. Identify affected columns and apply an appropriate "
                      "imputation strategy (median for numeric, mode for categorical). Wrap in a "
                      "sklearn Pipeline to prevent leakage.",
            code_snippet=(
                "from sklearn.impute import SimpleImputer\n"
                "from sklearn.pipeline import Pipeline\n"
                "imputer = SimpleImputer(strategy='median')\n"
                "pipe = Pipeline([('imputer', imputer), ('model', your_model)])"
            ),
        )]
    high_missing = [c for c, rate in flagged.items() if rate > 0.5]
    low_missing  = [c for c, rate in flagged.items() if rate <= 0.5]
    recs = []
    if high_missing:
        recs.append(Recommendation(
            check_name="missing_values",
            priority="high",
            action=f"Drop columns with > 50 % missing: {high_missing}",
            rationale="Columns missing more than half their values provide more noise than signal "
                      "and may destabilise imputation.",
            code_snippet=(
                f"cols_to_drop = {high_missing}\n"
                "df.drop(columns=cols_to_drop, inplace=True)"
            ),
        ))
    if low_missing:
        recs.append(Recommendation(
            check_name="missing_values",
            priority="medium",
            action=f"Impute moderate missing values in: {low_missing}",
            rationale="Moderate missingness can be handled with median imputation for numeric "
                      "features or most-frequent for categoricals. Use a pipeline to prevent "
                      "leakage between train and test.",
            code_snippet=(
                "from sklearn.impute import SimpleImputer\n"
                "from sklearn.pipeline import Pipeline\n"
                "imputer = SimpleImputer(strategy='median')  # or 'most_frequent'\n"
                "# Wrap in Pipeline so imputer is fit only on training data\n"
                "pipe = Pipeline([('imputer', imputer), ('model', your_model)])"
            ),
        ))
    return recs


def _rec_duplicates(r: CheckResult) -> list[Recommendation]:
    return [Recommendation(
        check_name="duplicates",
        priority="medium",
        action="Remove duplicate rows before splitting into train/test",
        rationale=(
            f"{r.details.get('duplicate_count', '?')} duplicate rows "
            f"({r.details.get('duplicate_rate', 0):.1%} of dataset) will cause "
            "identical samples in both train and test sets, inflating CV scores."
        ),
        code_snippet=(
            "df.drop_duplicates(inplace=True)\n"
            "df.reset_index(drop=True, inplace=True)"
        ),
    )]


def _rec_outliers(r: CheckResult) -> list[Recommendation]:
    flagged = r.details.get("flagged_columns", {})
    return [Recommendation(
        check_name="outliers",
        priority="medium",
        action=f"Investigate and cap outliers in: {list(flagged.keys())}",
        rationale="Extreme values can distort linear models and distance-based algorithms. "
                  "Tree-based models are robust to outliers but extreme values may still indicate "
                  "data collection errors.",
        code_snippet=(
            "# Option 1: Cap at IQR bounds\n"
            "for col in flagged_cols:\n"
            "    Q1, Q3 = df[col].quantile([0.25, 0.75])\n"
            "    IQR = Q3 - Q1\n"
            "    df[col] = df[col].clip(Q1 - 3*IQR, Q3 + 3*IQR)\n\n"
            "# Option 2: Use a robust scaler in the pipeline\n"
            "from sklearn.preprocessing import RobustScaler\n"
            "scaler = RobustScaler()  # uses median and IQR — resistant to outliers"
        ),
    )]


def _rec_class_imbalance(r: CheckResult) -> list[Recommendation]:
    ratio = r.details.get("minority_ratio", 0)
    return [Recommendation(
        check_name="class_imbalance",
        priority="high" if ratio < 0.05 else "medium",
        action="Address class imbalance before training",
        rationale=(
            f"Minority class represents only {ratio:.1%} of samples. "
            "Accuracy will be misleading — a model predicting only the majority class "
            "achieves high accuracy while being useless."
        ),
        code_snippet=(
            "# Option 1: Class-weighted loss (zero extra cost)\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "model = LogisticRegression(class_weight='balanced')\n\n"
            "# Option 2: Oversample minority class with SMOTE\n"
            "from imblearn.over_sampling import SMOTE\n"
            "X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)\n\n"
            "# Option 3: Use F1-macro or ROC-AUC instead of accuracy\n"
            "from sklearn.model_selection import cross_val_score\n"
            "scores = cross_val_score(model, X, y, scoring='f1_macro', cv=5)"
        ),
    )]


def _rec_constant_features(r: CheckResult) -> list[Recommendation]:
    cols = r.details.get("constant_columns", [])
    return [Recommendation(
        check_name="constant_features",
        priority="low",
        action=f"Drop constant columns: {cols}",
        rationale="Constant columns carry zero information for any model and may cause "
                  "numerical issues in some algorithms (e.g. division by zero in scaling).",
        code_snippet=(
            f"df.drop(columns={cols}, inplace=True)"
        ),
    )]


def _rec_low_variance(r: CheckResult) -> list[Recommendation]:
    cols = list(r.details.get("flagged_columns", {}).keys())
    return [Recommendation(
        check_name="low_variance",
        priority="low",
        action=f"Evaluate near-constant columns for removal: {cols}",
        rationale="Near-constant features contribute negligible information and may amplify "
                  "noise during training, especially in linear models.",
        code_snippet=(
            "from sklearn.feature_selection import VarianceThreshold\n"
            "# Remove features with CV < 1 % (threshold on normalised variance)\n"
            "selector = VarianceThreshold(threshold=0.001)\n"
            "X_filtered = selector.fit_transform(X)"
        ),
    )]


def _rec_target_leakage(r: CheckResult) -> list[Recommendation]:
    features = list(r.details.get("flagged_features", {}).keys())
    return [Recommendation(
        check_name="target_leakage",
        priority="high",
        action=f"Remove leaky features immediately: {features}",
        rationale=(
            "These features are almost perfectly correlated with the target — they are "
            "either derived from the outcome or are recorded after it. Keeping them will "
            "produce inflated validation scores that collapse in production."
        ),
        code_snippet=(
            f"leaky_cols = {features}\n"
            "df_clean = df.drop(columns=leaky_cols)\n\n"
            "# Verify impact: retrain without them and compare CV score\n"
            "# A large accuracy drop confirms the leakage was real."
        ),
    )]


def _rec_train_test_overlap(r: CheckResult) -> list[Recommendation]:
    return [Recommendation(
        check_name="train_test_overlap",
        priority="high",
        action="Deduplicate dataset before any train/test split",
        rationale=(
            f"{r.details.get('overlap_rows', '?')} rows appear in both simulated splits. "
            "The model sees test samples during training, giving unrealistically optimistic scores."
        ),
        code_snippet=(
            "df.drop_duplicates(inplace=True)\n"
            "df.reset_index(drop=True, inplace=True)\n"
            "# Then split\n"
            "from sklearn.model_selection import train_test_split\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42, stratify=y)"
        ),
    )]


def _rec_temporal_leakage(r: CheckResult) -> list[Recommendation]:
    date_col = r.details.get("date_column", "date")
    return [Recommendation(
        check_name="temporal_leakage",
        priority="high",
        action=f"Sort by '{date_col}' and use time-aware cross-validation",
        rationale="A random train/test split on unsorted temporal data mixes future observations "
                  "into training, allowing the model to learn patterns that would not exist "
                  "at inference time.",
        code_snippet=(
            f"df = df.sort_values('{date_col}').reset_index(drop=True)\n\n"
            "# Use TimeSeriesSplit instead of random K-Fold\n"
            "from sklearn.model_selection import TimeSeriesSplit\n"
            "tscv = TimeSeriesSplit(n_splits=5)\n"
            "scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')"
        ),
    )]


def _rec_id_column_leakage(r: CheckResult) -> list[Recommendation]:
    cols = list(r.details.get("flagged_columns", {}).keys())
    return [Recommendation(
        check_name="id_column_leakage",
        priority="medium",
        action=f"Drop identifier columns from the feature set: {cols}",
        rationale="High-cardinality identifier columns enable the model to memorise training "
                  "samples instead of learning generalisable patterns. They provide no signal "
                  "for unseen data.",
        code_snippet=(
            f"id_cols = {cols}\n"
            "X = df.drop(columns=id_cols + [target_col])"
        ),
    )]


def _rec_feature_correlation(r: CheckResult) -> list[Recommendation]:
    pairs = r.details.get("correlated_pairs", [])
    return [Recommendation(
        check_name="feature_correlation",
        priority="medium",
        action=f"Address {len(pairs)} highly correlated feature pair(s)",
        rationale="Highly correlated features carry redundant information, increase model "
                  "training time, can destabilise coefficient estimation in linear models, "
                  "and make feature importance harder to interpret.",
        code_snippet=(
            "# Option 1: Drop one feature from each correlated pair\n"
            "import pandas as pd\n"
            "corr_matrix = df.corr().abs()\n"
            "upper = corr_matrix.where(pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool))\n"
            "to_drop = [col for col in upper.columns if any(upper[col] > 0.90)]\n"
            "df.drop(columns=to_drop, inplace=True)\n\n"
            "# Option 2: Apply PCA to compress correlated features\n"
            "from sklearn.decomposition import PCA\n"
            "pca = PCA(n_components=0.95)  # keep 95% of variance\n"
            "X_pca = pca.fit_transform(X)"
        ),
    )]


def _rec_feature_relevance(r: CheckResult) -> list[Recommendation]:
    cols = r.affected_columns
    return [Recommendation(
        check_name="feature_relevance",
        priority="medium",
        action=f"Remove or investigate low-relevance features: {cols}",
        rationale="Features with near-zero mutual information with the target are likely noise. "
                  "Removing them reduces overfitting risk and speeds up training.",
        code_snippet=(
            "from sklearn.feature_selection import mutual_info_classif, SelectKBest\n\n"
            "# Select top-k features by mutual information\n"
            "selector = SelectKBest(mutual_info_classif, k='all')\n"
            "selector.fit(X_train, y_train)\n"
            "mi_scores = dict(zip(X_train.columns, selector.scores_))\n"
            "features_to_keep = [f for f, s in mi_scores.items() if s > 0.01]\n"
            "X_train = X_train[features_to_keep]"
        ),
    )]


def _rec_distribution_shape(r: CheckResult) -> list[Recommendation]:
    cols = r.affected_columns
    return [Recommendation(
        check_name="distribution_shape",
        priority="low",
        action=f"Apply variance-stabilising transforms to skewed features: {cols}",
        rationale="Highly skewed distributions can reduce the effectiveness of linear models "
                  "and distance-based algorithms. Transforming them towards normality often "
                  "improves model performance.",
        code_snippet=(
            "from sklearn.preprocessing import PowerTransformer\n"
            "import numpy as np\n\n"
            "# Yeo-Johnson handles positive and negative values\n"
            "pt = PowerTransformer(method='yeo-johnson')\n"
            f"df[{cols}] = pt.fit_transform(df[{cols}])\n\n"
            "# Or: simple log1p for right-skewed positive features\n"
            "df['feature'] = np.log1p(df['feature'])"
        ),
    )]


def _rec_sample_size(r: CheckResult) -> list[Recommendation]:
    n = r.details.get("n_rows", "?")
    return [Recommendation(
        check_name="sample_size",
        priority="high" if r.severity == "error" else "medium",
        action="Collect more data or apply data augmentation",
        rationale=(
            f"Dataset has {n} rows. Small datasets produce unreliable CV estimates, "
            "increase overfitting risk, and make per-class metrics unstable."
        ),
        code_snippet=(
            "# Option 1: Oversample with SMOTE (classification)\n"
            "from imblearn.over_sampling import SMOTE\n"
            "X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)\n\n"
            "# Option 2: Augment with noise for tabular data\n"
            "noise = pd.DataFrame(X.values + 0.01 * np.random.randn(*X.shape),\n"
            "                     columns=X.columns)\n"
            "X_aug = pd.concat([X, noise]).reset_index(drop=True)"
        ),
    )]


def _rec_class_support(r: CheckResult) -> list[Recommendation]:
    insufficient = r.details.get("insufficient_classes", {})
    return [Recommendation(
        check_name="class_support",
        priority="high",
        action=f"Increase support for under-represented classes: {list(insufficient.keys())}",
        rationale="Classes with very few samples produce unreliable per-class metrics and "
                  "may be entirely absent from some CV folds, causing errors.",
        code_snippet=(
            "from imblearn.over_sampling import SMOTE\n"
            "X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)\n\n"
            "# Or: use stratified k-fold to ensure class representation\n"
            "from sklearn.model_selection import StratifiedKFold\n"
            "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"
        ),
    )]


def _rec_cv_stability(r: CheckResult) -> list[Recommendation]:
    return [Recommendation(
        check_name="cv_stability",
        priority="medium",
        action="Reduce CV variance by increasing dataset size or using repeated k-fold",
        rationale="High CV standard deviation means performance estimates are unreliable — "
                  "the true model performance could differ significantly from the reported score.",
        code_snippet=(
            "from sklearn.model_selection import RepeatedStratifiedKFold\n"
            "# Repeat 10 times to get more stable estimates\n"
            "cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)\n"
            "scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')\n"
            "print(f'Mean: {scores.mean():.3f} ± {scores.std():.3f}')"
        ),
    )]


def _rec_feature_to_sample_ratio(r: CheckResult) -> list[Recommendation]:
    ratio = r.details.get("p_n_ratio", "?")
    return [Recommendation(
        check_name="feature_to_sample_ratio",
        priority="high" if r.severity == "error" else "medium",
        action=f"Reduce feature count (p/n={ratio}) via feature selection or PCA",
        rationale="Too many features relative to samples causes overfitting — the model "
                  "memorises training noise instead of learning generalisable patterns.",
        code_snippet=(
            "# Option 1: Select top-k features by importance\n"
            "from sklearn.feature_selection import SelectFromModel\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "selector = SelectFromModel(RandomForestClassifier(n_estimators=50))\n"
            "X_sel = selector.fit_transform(X, y)\n\n"
            "# Option 2: Dimensionality reduction with PCA\n"
            "from sklearn.decomposition import PCA\n"
            "pca = PCA(n_components=0.95)  # keep 95 % of variance\n"
            "X_pca = pca.fit_transform(X)"
        ),
    )]


def _rec_covariate_drift(r: CheckResult) -> list[Recommendation]:
    features = r.affected_columns
    return [Recommendation(
        check_name="covariate_drift",
        priority="high",
        action=f"Investigate distribution shift in: {features}",
        rationale="Different feature distributions in different data periods mean the model "
                  "trained on early data may not generalise to recent data.",
        code_snippet=(
            "# Diagnose with time-based split\n"
            "df_train = df[df[date_col] < split_date]\n"
            "df_test  = df[df[date_col] >= split_date]\n\n"
            "# Compute PSI for each feature\n"
            "for col in numeric_cols:\n"
            "    psi = compute_psi(df_train[col], df_test[col])\n"
            "    print(f'{col}: PSI = {psi:.3f}')"
        ),
    )]


def _rec_label_drift(r: CheckResult) -> list[Recommendation]:
    return [Recommendation(
        check_name="label_drift",
        priority="high",
        action="Investigate target distribution shift across time periods",
        rationale="A changing label distribution means the underlying phenomenon being "
                  "predicted has changed — retraining on recent data only may be necessary.",
        code_snippet=(
            "# Re-weight training samples by recency\n"
            "days_old = (df[date_col].max() - df[date_col]).dt.days\n"
            "weights  = np.exp(-days_old / 365)  # exponential decay\n"
            "model.fit(X_train, y_train, sample_weight=weights_train)"
        ),
    )]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_RECOMMENDATION_MAP = {
    # Quality
    "missing_values":           _rec_missing_values,
    "duplicates":               _rec_duplicates,
    "outliers":                 _rec_outliers,
    "class_imbalance":          _rec_class_imbalance,
    "constant_features":        _rec_constant_features,
    "low_variance":             _rec_low_variance,
    # Leakage
    "target_leakage":           _rec_target_leakage,
    "train_test_overlap":       _rec_train_test_overlap,
    "temporal_leakage":         _rec_temporal_leakage,
    "id_column_leakage":        _rec_id_column_leakage,
    # Feature analysis
    "feature_correlation":      _rec_feature_correlation,
    "feature_relevance":        _rec_feature_relevance,
    "distribution_shape":       _rec_distribution_shape,
    # Sufficiency
    "sample_size":              _rec_sample_size,
    "class_support":            _rec_class_support,
    "cv_stability":             _rec_cv_stability,
    "feature_to_sample_ratio":  _rec_feature_to_sample_ratio,
    # Drift
    "covariate_drift":          _rec_covariate_drift,
    "label_drift":              _rec_label_drift,
}

_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def generate_recommendations(report: FrameworkReport) -> list[Recommendation]:
    """Generate actionable recommendations for every failed check in *report*.

    Args:
        report: Populated FrameworkReport from all previous phases.

    Returns:
        List of :class:`~src.utils.Recommendation` sorted by priority
        (high → medium → low).
    """
    recs: list[Recommendation] = []
    all_results = (
        report.quality_results
        + report.leakage_results
        + report.feature_results
        + report.sufficiency_results
        + report.drift_results
    )

    for result in all_results:
        if result.passed:
            continue
        fn = _RECOMMENDATION_MAP.get(result.check_name)
        if fn:
            recs.extend(fn(result))
        else:
            logger.debug(f"No recommendation handler for check '{result.check_name}'")

    recs.sort(key=lambda r: _PRIORITY_ORDER.get(r.priority, 3))
    logger.info(
        f"Recommendations generated: {len(recs)} "
        f"({sum(1 for r in recs if r.priority=='high')} high, "
        f"{sum(1 for r in recs if r.priority=='medium')} medium, "
        f"{sum(1 for r in recs if r.priority=='low')} low)"
    )
    return recs
