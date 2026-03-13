from src.features import tfidf, embeddings


FEATURE_METHODS = {
    "tfidf": tfidf,
    "embeddings": embeddings,
}


def get_feature_module(method: str):
    method = method.lower().strip()

    if method not in FEATURE_METHODS:
        available = ", ".join(FEATURE_METHODS.keys())
        raise ValueError(
            f"Unknown feature method '{method}'. Available methods: {available}"
        )

    return FEATURE_METHODS[method]