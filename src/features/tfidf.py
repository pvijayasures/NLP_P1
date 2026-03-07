from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from src.config import TFIDF_CONFIG, VECTORIZER_FILE


def build_vectorizer():
    """Create a TF-IDF vectorizer from config."""
    return TfidfVectorizer(**TFIDF_CONFIG)


def fit_vectorizer(vectorizer, texts):
    """Fit vectorizer on training texts."""
    return vectorizer.fit(texts)


def transform_text(vectorizer, texts):
    """Transform texts to TF-IDF features."""
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer):
    """Save vectorizer to disk."""
    joblib.dump(vectorizer, VECTORIZER_FILE)


def load_vectorizer():
    """Load vectorizer from disk."""
    return joblib.load(VECTORIZER_FILE)