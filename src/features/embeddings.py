import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_FILE


def build_vectorizer():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def fit_vectorizer(vectorizer, texts):
    return vectorizer


def transform_text(vectorizer, texts):
    texts = list(texts.fillna("").astype(str))
    embeddings = vectorizer.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings)


def fit_transform_text(vectorizer, texts):
    vectorizer = fit_vectorizer(vectorizer, texts)
    return transform_text(vectorizer, texts)


def save_vectorizer(vectorizer):
    EMBEDDING_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model_name": EMBEDDING_MODEL_NAME}, EMBEDDING_MODEL_FILE)


def load_vectorizer():
    metadata = joblib.load(EMBEDDING_MODEL_FILE)
    return SentenceTransformer(metadata["model_name"])