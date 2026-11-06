"""
Student Name: Kieren Roberts
Student FAN: robe0980
File: train_models.py
Date: 3/11/2026
Description: Reads the data contract, selects leakage-free features, then
             trains, evaluates and compares any_fsi classifiers.
License: MIT
Usage: python3 train_models.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# ---------------------------------------------------------
# 1. File locations and settings
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

CONTRACT_FILE = PROJECT_DIR / "crash-risk-ingestion" / "docs" / "data_contract.json"
DATA_FILE = BASE_DIR / "processed" / "intersection_sites.csv"
MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "output"

# Table in the contract that DATA_FILE corresponds to.
TABLE = "intersection_sites"
TARGET = "any_fsi"

RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5

# Cross-validated metric used to rank the models and choose the best one.
# ROC-AUC does not depend on a decision threshold, which suits a small,
# imbalanced dataset.
SELECTION_METRIC = "roc_auc"


# ---------------------------------------------------------
# 2. Excluded features
# ---------------------------------------------------------

# any_fsi is 1 when the site had at least one fatal or serious injury
# crash. Every column built from the crash records (counts, casualties,
# injuries, crash types, rates) either encodes the target directly or is
# only known after the crashes happened, so all of them are excluded.
# Prefix rules catch these columns even if the contract adds more later.
LEAKAGE_PREFIXES = (
    "cse_",     # crash severity counts (pdo, inj, si, fat, minor_inj)
    "cty_",     # crash type counts
    "uty_",     # road-user type counts (bicycle, pedestrian)
    "lco_",     # lighting condition counts
    "total_",   # total crashes / casualties / fatalities / serious injuries
    "fsi_",     # fsi_crashes, fsi_rate_per_mev
    "crash_",   # crash_rate_per_mev
)

LEAKAGE_COLUMNS = {
    "any_fsi",
    "max_severity",
    "max_severity_label",
    "n_crash_locations",   # number of crash points matched to the site
    "night_share",         # derived from lco_night / total_crashes
    "vru_involved",        # derived from uty_* counts
}

# Identifiers and free text: unique per site, so they cannot generalise.
IDENTIFIER_COLUMNS = {
    "site_key",
    "site_id",
    "site_desc",
    "unique_loc",
    "segment_id",
    "intersection_site_key",
    "geometry",
}

# Exposure columns that are an exact rescaling of another feature
# (million_entering_vehicles = entering_aadt x 365 x 5 / 1e6).
REDUNDANT_COLUMNS = {
    "million_entering_vehicles",
}

# Contract dtypes that are treated as categorical features.
CATEGORICAL_DTYPES = {"str", "string", "object", "category"}


def is_excluded(column):
    """Return the reason a column is excluded, or None if it is usable."""

    if column in LEAKAGE_COLUMNS or column.startswith(LEAKAGE_PREFIXES):
        return "outcome / leakage"
    if column in IDENTIFIER_COLUMNS:
        return "identifier"
    if column in REDUNDANT_COLUMNS:
        return "redundant"
    return None


# ---------------------------------------------------------
# 3. Load and validate data
# ---------------------------------------------------------

def load_contract():
    """Load the table definition for TABLE from the data contract."""

    if not CONTRACT_FILE.is_file():
        raise FileNotFoundError(f"Data contract not found: {CONTRACT_FILE}")

    contract = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
    tables = contract.get("tables", {})

    if TABLE not in tables:
        raise ValueError(
            f"Table {TABLE!r} is not in the data contract. "
            f"Available tables: {', '.join(tables)}"
        )

    table = tables[TABLE]
    print(f"Data contract v{contract.get('contract_version')} "
          f"({contract.get('crash_period')}), table: {TABLE}")
    print(f"Grain: {table.get('grain')}")
    return table


def select_features(table):
    """Pick numeric and categorical features from the contract columns."""

    expected_rows = table.get("rows")
    numeric, categorical, excluded = [], [], {}

    for column in table["columns"]:
        name = column["name"]
        reason = is_excluded(name)

        # A column that is entirely null in the contract carries no signal.
        if reason is None and expected_rows and column.get("nulls") == expected_rows:
            reason = "all null"

        if reason:
            excluded[name] = reason
        elif column["dtype"] in CATEGORICAL_DTYPES:
            categorical.append(name)
        else:
            numeric.append(name)

    return numeric, categorical, excluded


def load_dataset(table, numeric, categorical):
    """Load DATA_FILE and check it against the contract."""

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            f"Check that {DATA_FILE.name} is in the processed folder."
        )

    data = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

    # Every contract column must be present.
    contract_columns = [column["name"] for column in table["columns"]]
    missing = [column for column in contract_columns if column not in data.columns]
    if missing:
        raise ValueError("Dataset is missing contract columns: " + ", ".join(missing))

    extra = [column for column in data.columns if column not in contract_columns]
    if extra:
        print("Warning: columns not in the contract are ignored:", ", ".join(extra))

    # Row count and primary key.
    expected_rows = table.get("rows")
    if expected_rows and len(data) != expected_rows:
        print(f"Warning: contract expects {expected_rows} rows, found {len(data)}.")

    primary_key = table.get("primary_key")
    if primary_key and data[primary_key].duplicated().any():
        raise ValueError(f"Primary key {primary_key!r} contains duplicates.")

    # Numeric features must be numeric; anything unparsable becomes NaN.
    for column in numeric:
        before = data[column].isna().sum()
        data[column] = pd.to_numeric(data[column], errors="coerce")
        coerced = data[column].isna().sum() - before
        if coerced:
            print(f"Warning: {coerced} non-numeric value(s) in {column} set to missing.")

    for column in categorical:
        data[column] = data[column].astype("object").where(data[column].notna(), np.nan)

    # Nulls above the contract's documented count suggest an upstream change.
    null_limits = {column["name"]: column.get("nulls", 0) for column in table["columns"]}
    for column in numeric + categorical:
        found = int(data[column].isna().sum())
        if found > null_limits.get(column, 0):
            print(f"Warning: {column} has {found} nulls (contract: {null_limits[column]}).")

    # Target: binary 0/1 with enough examples of each class for CV.
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET]).copy()

    if not data[TARGET].isin([0, 1]).all():
        raise ValueError(f"{TARGET} must contain only 0 or 1.")

    data[TARGET] = data[TARGET].astype(int)
    counts = data[TARGET].value_counts()

    if len(counts) != 2:
        raise ValueError("Training requires examples of both target classes.")
    if counts.min() < CV_FOLDS:
        raise ValueError(
            f"Each target class needs at least {CV_FOLDS} examples for "
            f"{CV_FOLDS}-fold cross-validation."
        )

    return data


def drop_constant_features(data, numeric, categorical, excluded):
    """Remove features with a single value in the data (e.g. site_type)."""

    for features in (numeric, categorical):
        for column in list(features):
            if data[column].nunique(dropna=True) <= 1:
                features.remove(column)
                excluded[column] = "constant"


# ---------------------------------------------------------
# 4. Preprocessing and pipeline
# ---------------------------------------------------------

def build_preprocessor(numeric, categorical):
    """Impute and scale numeric features; impute and one-hot categoricals."""

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])


def build_models(y_train):
    """Candidate classifiers. Each handles the class imbalance."""

    negatives, positives = np.bincount(y_train)

    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=negatives / positives,
            eval_metric="logloss",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
    }


def build_pipeline(model, numeric, categorical):
    """Chain the preprocessor and a classifier into one saved object."""

    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(numeric, categorical)),
        ("classifier", model),
    ])


# ---------------------------------------------------------
# 5. Iterate over and evaluate models
# ---------------------------------------------------------

CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]


def cross_validate_model(pipeline, X_train, y_train):
    """Stratified k-fold CV on the training split only."""

    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        scores = cross_validate(
            pipeline, X_train, y_train,
            cv=folds, scoring=CV_SCORING, n_jobs=-1,
        )

    return {
        f"cv_{metric}_{stat}": float(getattr(scores[f"test_{metric}"], stat)())
        for metric in CV_SCORING
        for stat in ("mean", "std")
    }


def test_metrics(pipeline, X_test, y_test):
    """Score a fitted pipeline on the held-out test split."""

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    return {
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_average_precision": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=["no_fsi", "any_fsi"],
            output_dict=True, zero_division=0,
        ),
    }


def evaluate_model(name, pipeline, split, features):
    """Cross-validate, fit, test and save one model and its metrics."""

    X_train, X_test, y_train, y_test = split
    print(f"\nTraining {name}...")

    cv_scores = cross_validate_model(pipeline, X_train, y_train)
    pipeline.fit(X_train, y_train)
    scores = test_metrics(pipeline, X_test, y_test)

    metrics = {
        "model": name,
        "target": TARGET,
        "table": TABLE,
        "features": features,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        **cv_scores,
        **scores,
    }

    model_path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(pipeline, model_path)

    metrics_path = METRICS_DIR / f"{name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"  {CV_FOLDS}-fold CV ROC-AUC: {cv_scores['cv_roc_auc_mean']:.3f} "
          f"+/- {cv_scores['cv_roc_auc_std']:.3f}")
    print(f"  {CV_FOLDS}-fold CV F1:      {cv_scores['cv_f1_mean']:.3f} "
          f"+/- {cv_scores['cv_f1_std']:.3f}")
    print(f"  Test ROC-AUC: {scores['test_roc_auc']:.3f}   "
          f"Test F1: {scores['test_f1']:.3f}   "
          f"Test recall: {scores['test_recall']:.3f}")
    print(f"  Confusion matrix: {scores['confusion_matrix']}")
    print(f"  Saved model:   {model_path}")
    print(f"  Saved metrics: {metrics_path}")

    return metrics


# ---------------------------------------------------------
# 6. Train and compare models
# ---------------------------------------------------------

COMPARISON_COLUMNS = [
    "model",
    "cv_roc_auc_mean", "cv_roc_auc_std",
    "cv_average_precision_mean", "cv_f1_mean", "cv_f1_std",
    "test_accuracy", "test_precision", "test_recall", "test_f1",
    "test_roc_auc", "test_average_precision",
]


def main():
    """Run the complete training and comparison process."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    table = load_contract()
    numeric, categorical, excluded = select_features(table)
    data = load_dataset(table, numeric, categorical)
    drop_constant_features(data, numeric, categorical, excluded)
    features = numeric + categorical

    if not features:
        raise ValueError("No usable features remain after exclusions.")

    print(f"\nRows with known target: {len(data)}")
    print("Target counts:")
    print(data[TARGET].value_counts().sort_index().to_string())

    print(f"\nPredictor features ({len(features)}):")
    for feature in features:
        kind = "categorical" if feature in categorical else "numeric"
        print(f"  - {feature} ({kind})")

    print(f"\nExcluded columns ({len(excluded)}):")
    for reason in sorted(set(excluded.values())):
        names = [name for name, why in excluded.items() if why == reason]
        print(f"  {reason}: {', '.join(names)}")

    X = data[features].copy()
    y = data[TARGET].copy()

    split = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    all_metrics = []
    for name, model in build_models(split[2]).items():
        pipeline = build_pipeline(model, numeric, categorical)
        all_metrics.append(evaluate_model(name, pipeline, split, features))

    # Rank on cross-validation (training data) rather than the single test
    # split, so the choice is not tuned to the test set.
    comparison = (
        pd.DataFrame(all_metrics)[COMPARISON_COLUMNS]
        .sort_values(f"cv_{SELECTION_METRIC}_mean", ascending=False)
        .reset_index(drop=True)
    )
    best_name = comparison.loc[0, "model"]

    joblib.dump(joblib.load(MODEL_DIR / f"{best_name}.joblib"),
                MODEL_DIR / "best_model.joblib")

    summary = {
        "table": TABLE,
        "target": TARGET,
        "selection_metric": f"cv_{SELECTION_METRIC}_mean",
        "best_model": best_name,
        "features": features,
        "excluded_features": excluded,
        "models": comparison.to_dict(orient="records"),
    }
    comparison_path = METRICS_DIR / "model_comparison.json"
    comparison_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nModel comparison (ranked by "
          f"{CV_FOLDS}-fold CV {SELECTION_METRIC})")
    print("-" * 60)
    print(comparison.round(3).to_string(index=False))
    print(f"\nBest model: {best_name} -> {MODEL_DIR / 'best_model.joblib'}")
    print(f"Comparison saved to: {comparison_path}")
    print("\nTraining and evaluation complete.")


if __name__ == "__main__":
    main()
