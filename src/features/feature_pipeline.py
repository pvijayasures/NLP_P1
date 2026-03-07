import pandas as pd

from src.config import (
    PROCESSED_PATH,
    TRAIN_PROCESSED_FILE,
    CLEAN_TEXT_COLUMN,
    LABEL_COLUMN,
)

from src.features.tfidf import (
    build_vectorizer,
    fit_vectorizer,
    transform_text,
    save_vectorizer,
)


def load_processed_data(filename: str = TRAIN_PROCESSED_FILE) -> pd.DataFrame:
    """Load a processed dataset from PROCESSED_PATH."""
    return pd.read_csv(PROCESSED_PATH / filename)


def prepare_features(filename: str = TRAIN_PROCESSED_FILE):
    """
    Prepare TF-IDF features from a processed dataset.

    Returns:
        X: sparse feature matrix
        y: labels
        vectorizer: fitted TF-IDF vectorizer
        df: loaded dataframe
    """
    df = load_processed_data(filename)

    texts = df[CLEAN_TEXT_COLUMN].fillna("")
    y = df[LABEL_COLUMN]

    vectorizer = build_vectorizer()
    vectorizer = fit_vectorizer(vectorizer, texts)
    X = transform_text(vectorizer, texts)

    save_vectorizer(vectorizer)

    return X, y, vectorizer, df