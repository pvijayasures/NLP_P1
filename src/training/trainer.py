import joblib

from sklearn.model_selection import train_test_split

from src.config import (
    TRAIN_PROCESSED_FILE,
    MODELS_PATH,
    TEST_SIZE,
    RANDOM_STATE,
)
from src.features import prepare_features
from src.models import get_model


def split_data(X, y):
    """Split data into train and test sets."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def train_model(X_train, y_train, model_name: str):
    """Train selected model."""
    model = get_model(model_name, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def save_model(model, model_name: str):
    """Save trained model."""
    model_file = MODELS_PATH / f"{model_name}_model.joblib"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file)
    return model_file


def run_training_pipeline(processed_filename: str, model_name: str):
    """Run feature preparation, splitting, training, and saving."""
    X, y, vectorizer, df = prepare_features(processed_filename)

    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train, model_name)
    model_file = save_model(model, model_name)

    return model, model_file, X_train, X_test, y_train, y_test, vectorizer, df