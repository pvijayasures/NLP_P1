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


def fit_transform_text(vectorizer, texts):
    """Fit vectorizer and transform texts."""
    return vectorizer.fit_transform(texts)


def save_vectorizer(vectorizer):
    """Save vectorizer to disk."""
    VECTORIZER_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving vectorizer to: {VECTORIZER_FILE}")
    joblib.dump(vectorizer, VECTORIZER_FILE)


def load_vectorizer():
    """Load vectorizer from disk."""
    print(f"Loading vectorizer from: {VECTORIZER_FILE}")
    return joblib.load(VECTORIZER_FILE)