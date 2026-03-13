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

MODELS_PATH = TRAINED_MODELS_PATH


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
# Feature engineering
# -------------------------------------------------------------------

FEATURE_METHOD = "tfidf"  # options: "tfidf", "embedding"

TFIDF_CONFIG = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
}


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_FILE = MODELS_PATH / "embedding_model.joblib"

TFIDF_VECTORIZER_FILE = VECTORIZERS_PATH / "tfidf_vectorizer.joblib"


# -------------------------------------------------------------------
# Training configuration
# -------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2


# -------------------------------------------------------------------
# Model selection
# -------------------------------------------------------------------

MODEL_NAME = "logreg"

# alternative models
# MODEL_NAME = "svm"
# MODEL_NAME = "naive_bayes"
# MODEL_NAME = "random_forest"


# -------------------------------------------------------------------
# Preprocessing configuration
# -------------------------------------------------------------------

# basic cleaning
LOWERCASE = True
REMOVE_URLS = True
REMOVE_HTML = True
REMOVE_NUMBERS = True
REMOVE_PUNCTUATION = True
NORMALIZE_WHITESPACE = True

# text normalization
NORMALIZE_CONTRACTIONS = False
NORMALIZE_PROFANITY = True
NORMALIZE_REPEATED_CHARACTERS = True

# token filtering
REMOVE_STOPWORDS = False
KEEP_NEGATIONS = True
REMOVE_SHORT_TOKENS = False
MIN_TOKEN_LENGTH = 2

# morphological normalization
STEM_WORDS = False
LEMMATIZE_WORDS = False

# dataset-level cleaning
DROP_DUPLICATES = True
DROP_EMPTY_TEXTS = True


# -------------------------------------------------------------------
# Regex patterns
# -------------------------------------------------------------------

URL_PATTERN = r"http\S+|www\S+"
HTML_PATTERN = r"<.*?>"
NUMBER_PATTERN = r"\d+"
WHITESPACE_PATTERN = r"\s+"

# token extraction
TOKEN_PATTERN = r"[a-zA-Z']+"


# -------------------------------------------------------------------
# Other text cleaning settings
# -------------------------------------------------------------------

PUNCTUATION = string.punctuation


# -------------------------------------------------------------------
# Stopword customization
# -------------------------------------------------------------------

NEGATION_WORDS = {
    "no",
    "not",
    "nor",
    "never",
    "cannot",
    "can't",
    "won't",
    "don't",
}

# -------------------------------------------------------------------
# Obfuscated profanity normalization
# -------------------------------------------------------------------

PROFANITY_PATTERNS = {
    r"\bf[\W_]*u[\W_]*c[\W_]*k+\b": "fuck",
    r"\bsh[\W_]*i[\W_]*t+\b": "shit",
    r"\bb[\W_]*i[\W_]*t[\W_]*c[\W_]*h+\b": "bitch",
    r"\ba[\W_]*s[\W_]*s[\W_]*h[\W_]*o[\W_]*l[\W_]*e\b": "asshole",
    r"\bi[\W_]*d[\W_]*i[\W_]*o[\W_]*t+\b": "idiot",
    r"\bm[\W_]*o[\W_]*r[\W_]*o[\W_]*n+\b": "moron",
}