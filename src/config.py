from pathlib import Path
import string

# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "raw"
INTERIM_PATH = DATA_DIR / "interim"
PROCESSED_PATH = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_PATH = MODELS_DIR / "trained"
VECTORIZERS_PATH = MODELS_DIR / "vectorizers"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_PATH = OUTPUTS_DIR / "metrics"
PLOTS_PATH = OUTPUTS_DIR / "plots"
PREDICTIONS_PATH = OUTPUTS_DIR / "predictions"
MODELS_PATH = MODELS_DIR / "trained"


# -------------------------------------------------------------------
# File names
# -------------------------------------------------------------------

TRAIN_RAW_FILE = "train.csv"
TEST_RAW_FILE = "test.csv"

TRAIN_INTERIM_FILE = "train_binary.csv"

TRAIN_PROCESSED_FILE = "train_binary_preprocessed.csv"
TEST_PROCESSED_FILE = "test_preprocessed.csv"


# -------------------------------------------------------------------
# Model / vectorizer artifacts
# -------------------------------------------------------------------

VECTORIZER_FILE = VECTORIZERS_PATH / "tfidf_vectorizer.joblib"


# -------------------------------------------------------------------
# Dataset columns
# -------------------------------------------------------------------

TEXT_COLUMN = "comment_text"
CLEAN_TEXT_COLUMN = "comment_text_clean"
LABEL_COLUMN = "label"


# -------------------------------------------------------------------
# Toxic classes (Jigsaw dataset)
# -------------------------------------------------------------------

TOXIC_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


# -------------------------------------------------------------------
# Feature extraction (TF-IDF)
# -------------------------------------------------------------------

TFIDF_CONFIG = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.9,
}


# -------------------------------------------------------------------
# Training configuration
# -------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2


# Default model
MODEL_NAME = "logreg"

# Alternative models
# MODEL_NAME = "svm"
# MODEL_NAME = "naive_bayes"
# MODEL_NAME = "random_forest"


# -------------------------------------------------------------------
# Preprocessing configuration
# -------------------------------------------------------------------

REMOVE_STOPWORDS = True
STEM_WORDS = True


# -------------------------------------------------------------------
# Regex patterns
# -------------------------------------------------------------------

URL_PATTERN = r"http\S+|www\S+"
HTML_PATTERN = r"<.*?>"
NUMBER_PATTERN = r"\d+"
WHITESPACE_PATTERN = r"\s+"
TOKEN_PATTERN = r"[a-z]+"


# -------------------------------------------------------------------
# Other text cleaning settings
# -------------------------------------------------------------------

PUNCTUATION = string.punctuation