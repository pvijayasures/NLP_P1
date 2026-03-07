import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    PROCESSED_PATH,
    CLEAN_TEXT_COLUMN,
    LABEL_COLUMN,
    TFIDF_CONFIG,
    TRAIN_PROCESSED_FILE,
)


def load_processed_file(filename: str) -> pd.DataFrame:
    """Load a processed CSV file."""
    return pd.read_csv(PROCESSED_PATH / filename)


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """Create TF-IDF vectorizer from config."""
    return TfidfVectorizer(**TFIDF_CONFIG)


def prepare_features(
    filename: str,
    text_column: str = CLEAN_TEXT_COLUMN,
    label_column: str = LABEL_COLUMN,
    vectorizer: TfidfVectorizer | None = None,
):
    """
    Prepare features from a processed dataset.

    - If vectorizer is None: fit + transform
    - If vectorizer is given: transform only
    """
    df = load_processed_file(filename)

    if text_column not in df.columns:
        raise ValueError(f"Missing text column: {text_column}")

    text_data = df[text_column].fillna("")
    y = df[label_column] if label_column in df.columns else None

    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer()
        X = vectorizer.fit_transform(text_data)
    else:
        X = vectorizer.transform(text_data)

    return X, y, vectorizer, df


if __name__ == "__main__":
    X_train, y_train, vectorizer, train_df = prepare_features(TRAIN_PROCESSED_FILE)
    print(f"Feature matrix shape: {X_train.shape}")