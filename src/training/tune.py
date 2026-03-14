"""Hyperparameter tuning for TF-IDF + Logistic Regression only."""

from __future__ import annotations

import json

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

from src.config import (
    METRICS_PATH,
    MODELS_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TRAIN_PROCESSED_FILE,
)
from src.features import prepare_features


INPUT_FILE = TRAIN_PROCESSED_FILE
HOLDOUT_TEST_SIZE = TEST_SIZE
SEED = RANDOM_STATE
CV_FOLDS = 5
SCORING = "f1"
N_JOBS = -1
VERBOSE = 1

# Stable grid: L2 only, no saga, no convergence issues in most cases
DEFAULT_PARAM_GRID = {
    "C": [0.1, 1.0, 3.0, 10.0],
    "solver": ["lbfgs"],
    "class_weight": ["balanced"],
    "max_iter": [2000, 5000],
    "tol": [1e-4, 1e-3],
}


def ensure_output_directories() -> None:
    """Create output folders if they do not exist yet."""
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Run hyperparameter tuning for the TF-IDF logistic regression model."""
    ensure_output_directories()

    print("=" * 60)
    print("Hyperparameter tuning: logreg + tfidf")
    print("=" * 60)

    X, y, _, _ = prepare_features(
        filename=INPUT_FILE,
        feature_method="tfidf",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=HOLDOUT_TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    grid_search = GridSearchCV(
        estimator=LogisticRegression(random_state=SEED),
        param_grid=DEFAULT_PARAM_GRID,
        scoring=SCORING,
        cv=CV_FOLDS,
        n_jobs=N_JOBS,
        verbose=VERBOSE,
        refit=True,
    )

    print("\nRunning grid search...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    results = {
        "model": "logreg",
        "feature_method": "tfidf",
        "input_file": INPUT_FILE,
        "cv_folds": CV_FOLDS,
        "scoring": SCORING,
        "best_params": grid_search.best_params_,
        "best_cv_score": float(grid_search.best_score_),
        "holdout_accuracy": report.get("accuracy", 0.0),
        "holdout_macro_f1": report.get("macro avg", {}).get("f1-score", 0.0),
        "holdout_weighted_f1": report.get("weighted avg", {}).get("f1-score", 0.0),
    }

    model_path = MODELS_PATH / "logreg_tfidf_tuned_model.joblib"
    metrics_path = METRICS_PATH / "logreg_tfidf_tuning_metrics.json"

    joblib.dump(best_model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nBest params:", grid_search.best_params_)
    print("Best CV score:", f"{grid_search.best_score_:.4f}")
    print("Holdout weighted F1:", f"{results['holdout_weighted_f1']:.4f}")
    print(f"Saved tuned model to: {model_path}")
    print(f"Saved tuning metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
