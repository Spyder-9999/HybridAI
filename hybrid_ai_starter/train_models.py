"""
File: train_models.py
Student: dobi0016
Course: COMP3742/9742 - Artificial Intelligence

Purpose:
Train and compare Logistic Regression and Decision Tree classifiers
for the intersection crash-history target any_fsi.

Important:
Crash counts, injury statistics, crash rates, and fields derived
from crash outcomes are deliberately excluded from the predictors.
This is an educational prototype, not a validated road-safety tool.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
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
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------
# 1. File locations and settings
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "processed" / "intersection_sites.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

TARGET = "any_fsi"
RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5


# ---------------------------------------------------------
# 2. Select leakage-conscious predictor features
# ---------------------------------------------------------

# These features describe the site and its context.
# Crash outcomes, injury counts, crash rates, and other
# outcome-derived columns are intentionally not included.

NUMERIC_FEATURES = [
    "n_points",
    "point_spread_m",
    "spread_flag",
    "entering_aadt",
    "major_road_aadt",
    "n_state_roads",
    "cv_percent",
    "volume_stale",
    "has_volume",
    "ped_crossings_200m",
    "school_crossings_300m",
    "dist_school_crossing_m",
    "dist_cbd_km",
    "lon",
    "lat",
]

CATEGORICAL_FEATURES = [
    "site_type",
    "road_no",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------
# 3. Load and validate the dataset
# ---------------------------------------------------------

def load_dataset():
    """Load the CSV and check that the required columns exist."""

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Check that intersection_sites.csv is in the processed folder."
        )

    data = pd.read_csv(DATA_FILE)

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    # Only rows with a known target can be used for supervised training.
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET]).copy()

    # The target must contain binary 0/1 values.
    invalid_targets = ~data[TARGET].isin([0, 1])

    if invalid_targets.any():
        raise ValueError(
            "The target column must contain only 0, 1, or missing values."
        )

    data[TARGET] = data[TARGET].astype(int)

    if data[TARGET].nunique() != 2:
        raise ValueError(
            "Training requires examples of both target classes."
        )

    if data[TARGET].value_counts().min() < CV_FOLDS:
        raise ValueError(
            f"Each target class needs at least {CV_FOLDS} examples "
            "for the selected cross-validation."
        )

    return data


# ---------------------------------------------------------
# 4. Build preprocessing and model pipelines
# ---------------------------------------------------------

def build_pipeline(model):
    """Create a pipeline that handles missing and categorical values."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


# ---------------------------------------------------------
# 5. Evaluate one model
# ---------------------------------------------------------

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train a model, evaluate it, and save its files."""

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Cross-validation uses the training data only.
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=CV_FOLDS,
        scoring="f1",
        n_jobs=-1,
    )

    metrics = {
        "model": name,
        "target": TARGET,
        "features": FEATURES,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "test_precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "test_recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "test_f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "test_roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "test_average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["no_fsi", "any_fsi"],
            output_dict=True,
            zero_division=0,
        ),
    }

    # Save the trained pipeline.
    model_path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, model_path)

    # Save a readable record of the evaluation.
    metrics_path = OUTPUT_DIR / f"{name}_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    print(f"Test precision: {metrics['test_precision']:.3f}")
    print(f"Test recall: {metrics['test_recall']:.3f}")
    print(f"Test F1-score: {metrics['test_f1']:.3f}")
    print(f"Test ROC-AUC: {metrics['test_roc_auc']:.3f}")
    print(
        "Test average precision: "
        f"{metrics['test_average_precision']:.3f}"
    )
    print(
        f"{CV_FOLDS}-fold CV F1: "
        f"{metrics['cv_f1_mean']:.3f} "
        f"+/- {metrics['cv_f1_std']:.3f}"
    )
    print("Confusion matrix:", metrics["confusion_matrix"])
    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)

    return metrics


# ---------------------------------------------------------
# 6. Train and compare both classifiers
# ---------------------------------------------------------

def main():
    """Run the complete model training and comparison process."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset()

    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    print("Intersection crash-history classification")
    print("------------------------------------------")
    print("Rows with known target:", len(data))
    print("Target counts:")
    print(y.value_counts().sort_index().to_string())
    print("\nPredictor features:")
    for feature in FEATURES:
        print("-", feature)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "logistic_regression": build_pipeline(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "decision_tree": build_pipeline(
            DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
    }

    all_metrics = {}

    for name, model in models.items():
        all_metrics[name] = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

    comparison = []

    for name, metrics in all_metrics.items():
        comparison.append(
            {
                "model": name,
                "test_accuracy": metrics["test_accuracy"],
                "test_precision": metrics["test_precision"],
                "test_recall": metrics["test_recall"],
                "test_f1": metrics["test_f1"],
                "test_roc_auc": metrics["test_roc_auc"],
                "test_average_precision": (
                    metrics["test_average_precision"]
                ),
                "cv_f1_mean": metrics["cv_f1_mean"],
                "cv_f1_std": metrics["cv_f1_std"],
            }
        )

    comparison_path = OUTPUT_DIR / "model_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nModel comparison")
    print("----------------")
    print(
        pd.DataFrame(comparison)
        .round(3)
        .to_string(index=False)
    )
    print("\nComparison saved to:", comparison_path)
    print("\nTraining and evaluation complete.")


if __name__ == "__main__":
    main()
