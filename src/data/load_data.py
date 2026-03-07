import pandas as pd

from src.config import (
    RAW_PATH,
    INTERIM_PATH,
    TRAIN_RAW_FILE,
    TRAIN_INTERIM_FILE,
    TOXIC_COLUMNS,
    LABEL_COLUMN,
)


def load_train() -> pd.DataFrame:
    """Load raw training data."""
    return pd.read_csv(RAW_PATH / TRAIN_RAW_FILE)


def create_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create one binary label and drop the original toxicity columns."""
    df = df.copy()
    df[LABEL_COLUMN] = (df[TOXIC_COLUMNS].sum(axis=1) > 0).astype(int)
    df = df.drop(columns=TOXIC_COLUMNS)
    return df


def prepare_dataset() -> pd.DataFrame:
    """Prepare train dataset and save it to interim."""
    df = load_train()
    df = create_binary_label(df)

    INTERIM_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTERIM_PATH / TRAIN_INTERIM_FILE, index=False)

    return df


if __name__ == "__main__":
    df = prepare_dataset()

    print(df.head())
    print("\nLabel distribution:")
    print(df[LABEL_COLUMN].value_counts())