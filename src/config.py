from pathlib import Path
import string

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_PATH = DATA_DIR / "raw"
INTERIM_PATH = DATA_DIR / "interim"
PROCESSED_PATH = DATA_DIR / "processed"


# file names
TRAIN_RAW_FILE = "train.csv"
TEST_RAW_FILE = "test.csv"

TRAIN_INTERIM_FILE = "train_binary.csv"

TRAIN_PROCESSED_FILE = "train_binary_preprocessed.csv"
TEST_PROCESSED_FILE = "test_preprocessed.csv"


TEXT_COLUMN = "comment_text"
CLEAN_TEXT_COLUMN = "comment_text_clean"
LABEL_COLUMN = "label"


TOXIC_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


TFIDF_CONFIG = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.9,
}


RANDOM_STATE = 42
TEST_SIZE = 0.2

REMOVE_STOPWORDS = True
STEM_WORDS = True


URL_PATTERN = r"http\S+|www\S+"
HTML_PATTERN = r"<.*?>"
NUMBER_PATTERN = r"\d+"
WHITESPACE_PATTERN = r"\s+"
TOKEN_PATTERN = r"[a-z]+"

PUNCTUATION = string.punctuation