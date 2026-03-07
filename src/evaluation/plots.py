from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

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