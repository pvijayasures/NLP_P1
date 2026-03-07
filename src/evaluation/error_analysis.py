from pathlib import Path

import pandas as pd

from src.config import OUTPUTS_DIR, CLEAN_TEXT_COLUMN


PREDICTIONS_PATH = OUTPUTS_DIR / "predictions"


def build_error_analysis_dataframe(
    model,
    X_test,
    y_test,
    df_test: pd.DataFrame,
    text_column: str = CLEAN_TEXT_COLUMN,
) -> pd.DataFrame:
    """
    Build a dataframe for error analysis with predictions and misclassification flags.
    Assumes df_test matches X_test / y_test row order.
    """
    preds = model.predict(X_test)

    result_df = df_test.copy().reset_index(drop=True)
    result_df["true_label"] = pd.Series(y_test).reset_index(drop=True)
    result_df["pred_label"] = pd.Series(preds).reset_index(drop=True)
    result_df["is_error"] = result_df["true_label"] != result_df["pred_label"]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        result_df["pred_confidence"] = probs.max(axis=1)

    if text_column not in result_df.columns:
        raise ValueError(f"Column '{text_column}' not found in df_test.")

    return result_df


def save_error_analysis(
    error_df: pd.DataFrame,
    model_name: str,
    only_errors: bool = True,
) -> Path:
    """Save error analysis dataframe to CSV."""
    PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)

    if only_errors:
        output_df = error_df[error_df["is_error"]].copy()
        filename = f"{model_name}_errors.csv"
    else:
        output_df = error_df.copy()
        filename = f"{model_name}_predictions_detailed.csv"

    output_path = PREDICTIONS_PATH / filename
    output_df.to_csv(output_path, index=False)

    print(f"Saved error analysis file: {output_path}")
    return output_path


def print_top_errors(
    error_df: pd.DataFrame,
    text_column: str = CLEAN_TEXT_COLUMN,
    n: int = 10,
):
    """Print a few misclassified examples."""
    errors = error_df[error_df["is_error"]].copy()

    if errors.empty:
        print("No misclassified samples found.")
        return

    print(f"\nTop {min(n, len(errors))} misclassified examples:\n")

    for i, (_, row) in enumerate(errors.head(n).iterrows(), start=1):
        print(f"Example {i}")
        print(f"Text: {row[text_column]}")
        print(f"True label: {row['true_label']}")
        print(f"Predicted label: {row['pred_label']}")
        if "pred_confidence" in row:
            print(f"Confidence: {row['pred_confidence']:.4f}")
        print("-" * 80)