"""
Main entry point for the hate speech experimental pipeline.

Pipeline steps:
1. Load processed data
2. Build features
3. Train selected model
4. Evaluate model
5. Save metrics, predictions, and plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    TRAIN_PROCESSED_FILE,
    CLEAN_TEXT_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_NAME,
    MODELS_PATH,
    METRICS_PATH,
    PREDICTIONS_PATH,
    PLOTS_PATH,
    VECTORIZERS_PATH
)

from src.features import prepare_features
from src.models import get_model
from src.evaluation import (
    compute_metrics,
    print_metrics,
    save_metrics,
    save_confusion_matrix,
    save_class_distribution,
    save_prediction_confidence_histogram,
    build_error_analysis_dataframe,
    save_error_analysis,
    print_top_errors,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the full NLP experimental pipeline."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Model to train (e.g. logreg, nb, rf, svm).",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=TRAIN_PROCESSED_FILE,
        help="Processed input CSV filename.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Test split size.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random state for reproducibility.",
    )

    return parser.parse_args()


def ensure_directories() -> None:
    """Create required output directories if they do not exist."""
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    VECTORIZERS_PATH.mkdir(parents=True, exist_ok=True)


def save_predictions(
    df_test: pd.DataFrame,
    predictions,
    output_path: Path,
) -> None:
    """Save validation predictions to CSV."""
    result_df = df_test.copy()
    result_df["prediction"] = predictions
    result_df.to_csv(output_path, index=False)


def main() -> None:
    """Run the full experimental pipeline."""
    args = parse_args()
    ensure_directories()

    model_name = args.model.lower()

    print("=" * 60)
    print("Starting NLP experimental pipeline")
    print("=" * 60)
    print(f"Model        : {model_name}")
    print(f"Input file   : {args.input_file}")
    print(f"Test size    : {args.test_size}")
    print(f"Random state : {args.random_state}")

    print("\n[1/5] Preparing features...")
    X, y, vectorizer, df = prepare_features(args.input_file)

    print("\n[2/5] Splitting data...")
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X,
        y,
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    print("\n[3/5] Training model...")
    model = get_model(model_name, random_state=args.random_state)
    model.fit(X_train, y_train)

    model_file = MODELS_PATH / f"{model_name}_model.joblib"
    joblib.dump(model, model_file)
    print(f"Saved model to: {model_file}")

    print("\n[4/5] Evaluating model...")
    metrics = compute_metrics(model, X_test, y_test)
    print_metrics(metrics)

    metrics_file = METRICS_PATH / f"{model_name}_metrics.json"
    save_metrics(metrics, metrics_file)
    print(f"Saved metrics to: {metrics_file}")

    predictions_file = PREDICTIONS_PATH / f"{model_name}_val_predictions.csv"
    save_predictions(df_test, metrics["predictions"], predictions_file)
    print(f"Saved predictions to: {predictions_file}")

    print("\n[5/5] Creating plots and error analysis...")

    confusion_matrix_file = save_confusion_matrix(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
    )
    print(f"Saved confusion matrix to: {confusion_matrix_file}")

    class_distribution_file = save_class_distribution(
        y=y,
        filename=f"class_distribution_{model_name}.png",
    )
    print(f"Saved class distribution plot to: {class_distribution_file}")

    confidence_plot_file = save_prediction_confidence_histogram(
        model=model,
        X_test=X_test,
        model_name=model_name,
    )
    if confidence_plot_file is not None:
        print(f"Saved prediction confidence histogram to: {confidence_plot_file}")

    error_df = build_error_analysis_dataframe(
        model=model,
        X_test=X_test,
        y_test=y_test,
        df_test=df_test,
        text_column=CLEAN_TEXT_COLUMN,
    )

    error_file = save_error_analysis(
        error_df=error_df,
        model_name=model_name,
        only_errors=True,
    )
    print(f"Saved error analysis to: {error_file}")

    print("\nTop classification errors:")
    print_top_errors(error_df)

    print("\n" + "=" * 60)
    print("Pipeline finished successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()