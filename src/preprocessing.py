import re
import pandas as pd
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from src.config import (
    INTERIM_PATH,
    PROCESSED_PATH,
    TEXT_COLUMN,
    CLEAN_TEXT_COLUMN,
    REMOVE_STOPWORDS,
    STEM_WORDS,
    URL_PATTERN,
    HTML_PATTERN,
    NUMBER_PATTERN,
    WHITESPACE_PATTERN,
    TOKEN_PATTERN,
    PUNCTUATION,
    TRAIN_INTERIM_FILE,
    TRAIN_PROCESSED_FILE,
    TEST_RAW_FILE,
    TEST_PROCESSED_FILE,
)


STOPWORDS = set(stopwords.words("english"))
STEMMER = SnowballStemmer("english")


def normalize_text(text: str) -> str:
    """Basic text normalization."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(URL_PATTERN, " ", text)
    text = re.sub(HTML_PATTERN, " ", text)
    text = re.sub(NUMBER_PATTERN, " ", text)
    text = text.translate(str.maketrans("", "", PUNCTUATION))
    text = re.sub(WHITESPACE_PATTERN, " ", text).strip()

    return text


def tokenize(text: str) -> list[str]:
    """Extract alphabetic tokens."""
    return re.findall(TOKEN_PATTERN, text)


def remove_stopwords_tokens(tokens: list[str]) -> list[str]:
    """Remove common stopwords."""
    return [token for token in tokens if token not in STOPWORDS]


def stem_tokens(tokens: list[str]) -> list[str]:
    """Apply Snowball stemming."""
    return [STEMMER.stem(token) for token in tokens]


def clean_text(text: str) -> str:
    """Full preprocessing pipeline for a single text."""
    text = normalize_text(text)
    tokens = tokenize(text)

    if REMOVE_STOPWORDS:
        tokens = remove_stopwords_tokens(tokens)

    if STEM_WORDS:
        tokens = stem_tokens(tokens)

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to a dataframe."""
    df = df.copy()
    df[CLEAN_TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").apply(clean_text)
    return df


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
    # preprocess_file(TEST_RAW_FILE, TEST_PROCESSED_FILE)