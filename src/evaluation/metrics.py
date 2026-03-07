import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(model, X_test, y_test) -> dict:
    """Compute evaluation metrics for a trained model."""

    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "classification_report": classification_report(
            y_test, preds, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "predictions": preds.tolist(),
    }

    return metrics


def print_metrics(metrics: dict):
    """Print evaluation metrics in readable format."""

    print("\n==============================")
    print("Evaluation Metrics")
    print("==============================\n")

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")

    print("\nClassification Report:\n")

    report = metrics["classification_report"]

    for label, values in report.items():
        if isinstance(values, dict):
            precision = values.get("precision", 0)
            recall = values.get("recall", 0)
            f1 = values.get("f1-score", 0)
            support = values.get("support", 0)

            print(
                f"{label:15} "
                f"P={precision:.3f} "
                f"R={recall:.3f} "
                f"F1={f1:.3f} "
                f"Support={support}"
            )


def save_metrics(metrics: dict, output_path):
    """Save metrics to JSON."""

    output_path = Path(output_path)

    serializable_metrics = dict(metrics)
    serializable_metrics.pop("predictions", None)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, indent=4)