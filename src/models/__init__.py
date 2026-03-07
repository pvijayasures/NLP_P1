from src.models.logistic_regression import build_model as build_logreg
from src.models.svm import build_model as build_svm
from src.models.naive_bayes import build_model as build_naive_bayes
from src.models.random_forest import build_model as build_random_forest


MODEL_REGISTRY = {
    "logreg": build_logreg,
    "svm": build_svm,
    "naive_bayes": build_naive_bayes,
    "random_forest": build_random_forest,
}


def get_model(model_name: str, random_state: int | None = None):
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{model_name}' not supported. Available models: {available}"
        )

    return MODEL_REGISTRY[model_name](random_state=random_state)