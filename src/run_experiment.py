from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

# Importing your project-specific constants and registries
from src.config import PROJECT_ROOT, RANDOM_STATE, TEST_SIZE, TRAIN_PROCESSED_FILE
from src.features.factory import FEATURE_METHODS
from src.main import validate_model_feature_combination
from src.models import MODEL_REGISTRY


@dataclass(frozen=True)
class ExperimentRun:
    model: str
    feature_method: str


def get_all_combinations() -> tuple[list[ExperimentRun], list[ExperimentRun]]:
    """Builds a list of all valid and invalid model/feature combinations."""
    planned_runs: list[ExperimentRun] = []
    skipped_runs: list[ExperimentRun] = []

    # Automatically fetch all keys from your registries
    models = sorted(MODEL_REGISTRY.keys())
    feature_methods = sorted(FEATURE_METHODS.keys())

    for feature_method in feature_methods:
        for model in models:
            run = ExperimentRun(model=model, feature_method=feature_method)
            try:
                # Check if this specific pair is valid according to your logic
                validate_model_feature_combination(
                    model_name=model,
                    feature_method=feature_method,
                )
                planned_runs.append(run)
            except ValueError:
                skipped_runs.append(run)

    return planned_runs, skipped_runs


def run_experiment(run: ExperimentRun) -> int:
    """Executes the experiment as a subprocess call to src.main."""
    command = [
        sys.executable,
        "-m",
        "src.main",
        "--model", run.model,
        "--feature-method", run.feature_method,
        "--input-file", TRAIN_PROCESSED_FILE,
        "--test-size", str(TEST_SIZE),
        "--random-state", str(RANDOM_STATE),
    ]

    print("\n" + "-" * 60)
    print(f"Executing: {run.model} + {run.feature_method}")
    print(f"Command: {' '.join(command)}")

    # cwd=PROJECT_ROOT ensures the paths are resolved correctly
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return result.returncode


def main() -> None:
    print("Initializing Automated Experiment Runner...")
    print(f"Defaults: Test Size={TEST_SIZE}, Random State={RANDOM_STATE}")

    planned_runs, skipped_runs = get_all_combinations()

    if skipped_runs:
        print(f"\nSkipping {len(skipped_runs)} incompatible combinations.")

    if not planned_runs:
        print("Error: No valid combinations found in registries.")
        return

    print("=" * 60)
    print(f"Total Planned Runs: {len(planned_runs)}")
    print("=" * 60)

    successes: list[ExperimentRun] = []
    failures: list[ExperimentRun] = []

    for run in planned_runs:
        exit_code = run_experiment(run)
        if exit_code == 0:
            successes.append(run)
        else:
            failures.append(run)
            print(f"FAILED: {run}")

    # Final Summary Report
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total Successful: {len(successes)}")
    print(f"Total Failed    : {len(failures)}")

    if failures:
        print("\nFailed Combinations:")
        for f in failures:
            print(f"  [!] {f.model} / {f.feature_method}")
        sys.exit(1)

    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()
