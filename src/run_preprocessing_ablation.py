from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_PATH,
    OUTPUTS_DIR,
    CLEAN_TEXT_COLUMN,
    LABEL_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    TFIDF_CONFIG,
)

# ---------------------------------------------------------
# Output paths
# ---------------------------------------------------------
ABLATION_OUTPUT_DIR = OUTPUTS_DIR / "preprocessing_ablation"
ABLATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Path to save the final performance comparison
SUMMARY_FILE = ABLATION_OUTPUT_DIR / "logreg_preprocessing_ablation_results.csv"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def list_processed_csv_files(processed_dir: Path) -> list[Path]:
    """
    Return all experiment CSV files (starting with 'exp_').
    Excludes the summary metadata file.
    """
    # Look specifically for files starting with 'exp_' as defined in the previous script
    files = sorted(list(processed_dir.glob("exp_*.csv")))
    return files


def build_logreg_model() -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="liblinear",
    )


def compute_binary_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_pos": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_pos": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_pos": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def evaluate_file(csv_path: Path) -> dict[str, Any]:
    """
    Train and evaluate TF-IDF + Logistic Regression on one processed CSV file.
    """
    df = pd.read_csv(csv_path)

    required_columns = {CLEAN_TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"File '{csv_path.name}' is missing columns: {sorted(missing_columns)}"
        )

    # Basic cleanup for evaluation
    df = df.dropna(subset=[CLEAN_TEXT_COLUMN, LABEL_COLUMN]).copy()
    df[CLEAN_TEXT_COLUMN] = df[CLEAN_TEXT_COLUMN].astype(str)
    df = df[df[CLEAN_TEXT_COLUMN].str.strip() != ""].copy()

    if len(df) < 10:  # Safety check for tiny slices
        raise ValueError(f"File '{csv_path.name}' has insufficient rows ({len(df)}).")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df[CLEAN_TEXT_COLUMN],
        df[LABEL_COLUMN],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COLUMN],
    )

    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = build_logreg_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = compute_binary_metrics(y_test, y_pred)

    # Clean up the variant name for the report (remove 'exp_' prefix)
    variant_display_name = csv_path.stem.replace("exp_", "")

    return {
        "variant": variant_display_name,
        "n_rows": len(df),
        "n_features": X_train.shape[1],
        **metrics,
    }


def main() -> None:
    processed_files = list_processed_csv_files(PROCESSED_PATH)

    if not processed_files:
        raise FileNotFoundError(
            f"No files starting with 'exp_' found in: {PROCESSED_PATH}. "
            "Make sure you ran the preprocessing experiment script first."
        )

    print(f"Found {len(processed_files)} experiment files.")
    results: list[dict[str, Any]] = []

    for csv_path in processed_files:
        try:
            result = evaluate_file(csv_path)
            results.append(result)
            print(f"[OK] {result['variant']:<25} F1: {result['f1_pos']:.4f}")
        except Exception as exc:
            print(f"[FAIL] {csv_path.name}: {exc}")

    if not results:
        return

    # Sort results by F1 score to see what worked best
    results_df = pd.DataFrame(results).sort_values(by="f1_pos", ascending=False)
    results_df.to_csv(SUMMARY_FILE, index=False)

    print("\n--- Ablation Results (Sorted by F1) ---")
    print(results_df[["variant", "f1_pos", "n_features", "accuracy"]].to_string(index=False))

    print(f"\nDetailed report saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()