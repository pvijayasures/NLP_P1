from pathlib import Path
import pandas as pd

from src.config import (
    INTERIM_PATH,
    PROCESSED_PATH,
    TRAIN_INTERIM_FILE,
    TRAIN_PROCESSED_FILE,
)

from src.preprocessing.clean_text import preprocess_dataframe


def preprocess_file(input_filename: str, output_filename: str | None = None) -> pd.DataFrame:
    """Preprocess any CSV from INTERIM_PATH and save to PROCESSED_PATH."""
    input_path = INTERIM_PATH / input_filename

    if output_filename is None:
        output_filename = f"{Path(input_filename).stem}_preprocessed.csv"

    output_path = PROCESSED_PATH / output_filename

    df = pd.read_csv(input_path)
    df = preprocess_dataframe(df)

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

    return df


if __name__ == "__main__":
    preprocess_file(TRAIN_INTERIM_FILE, TRAIN_PROCESSED_FILE)