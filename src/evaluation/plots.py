from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import learning_curve

from src.config import OUTPUTS_DIR


PLOTS_PATH = OUTPUTS_DIR / "plots"


def save_confusion_matrix(model, X_test, y_test, model_name: str) -> Path:
    """Save confusion matrix plot."""
    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)

    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_PATH / f"confusion_matrix_{model_name}.png"

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved confusion matrix: {output_path}")
    return output_path


def save_class_distribution(y, filename: str = "class_distribution.png") -> Path:
    """Save class distribution plot."""
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_PATH / filename

    y_series = pd.Series(y)
    counts = y_series.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved class distribution plot: {output_path}")
    return output_path


def save_prediction_confidence_histogram(model, X_test, model_name: str) -> Path | None:
    """
    Save prediction confidence histogram if the model supports predict_proba.
    Returns None if unsupported.
    """
    if not hasattr(model, "predict_proba"):
        print(f"Model '{model_name}' does not support predict_proba. Skipping confidence plot.")
        return None

    probs = model.predict_proba(X_test)
    max_probs = probs.max(axis=1)

    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_PATH / f"prediction_confidence_{model_name}.png"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(max_probs, bins=20)
    ax.set_title(f"Prediction Confidence - {model_name}")
    ax.set_xlabel("Max Predicted Probability")
    ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved prediction confidence plot: {output_path}")
    return output_path


def save_learning_curve(
    model,
    X_train,
    y_train,
    model_name: str,
    cv: int = 5,
    scoring: str = "f1_weighted",
    train_sizes: list | None = None,
) -> Path:
    """Save a learning curve plot for the given model.

    Args:
        model: Sklearn-compatible estimator.
        X_train: Training feature matrix.
        y_train: Training labels.
        model_name: Name used for the output filename.
        cv: Number of cross-validation folds.
        scoring: Scoring metric passed to learning_curve.
        train_sizes: Relative or absolute training set sizes.

    Returns:
        Path to the saved plot.
    """
    if train_sizes is None:
        train_sizes = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        train_sizes=train_sizes,
        n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_PATH / f"learning_curve_{model_name}.png"

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(train_sizes_abs, train_mean, marker="o", label="Training score")
    ax.fill_between(
        train_sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
    )

    ax.plot(train_sizes_abs, val_mean, marker="s", label="Validation score")
    ax.fill_between(
        train_sizes_abs,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.2,
    )

    ax.set_title(f"Learning Curve – {model_name}")
    ax.set_xlabel("Training set size")
    ax.set_ylabel(scoring)
    ax.legend(loc="lower right")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved learning curve plot: {output_path}")
    return output_path
