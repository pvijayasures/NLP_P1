import pandas as pd

from src.config import (
    PROCESSED_PATH,
    TRAIN_PROCESSED_FILE,
    CLEAN_TEXT_COLUMN,
    LABEL_COLUMN,
    FEATURE_METHOD,
)

from src.features.factory import get_feature_module


def load_processed_data(filename: str = TRAIN_PROCESSED_FILE) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_PATH / filename)


def prepare_features(
    filename: str = TRAIN_PROCESSED_FILE,
    feature_method: str = FEATURE_METHOD,
):
    df = load_processed_data(filename)

    texts = df[CLEAN_TEXT_COLUMN].fillna("")
    y = df[LABEL_COLUMN]

    feature_module = get_feature_module(feature_method)

    vectorizer = feature_module.build_vectorizer()
    vectorizer = feature_module.fit_vectorizer(vectorizer, texts)
    X = feature_module.transform_text(vectorizer, texts)

    feature_module.save_vectorizer(vectorizer)

    return X, y, vectorizer, df