import re
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from src.config import (
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