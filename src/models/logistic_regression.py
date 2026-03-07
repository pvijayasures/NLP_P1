from sklearn.linear_model import LogisticRegression


def build_model(random_state: int | None = None):
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state
    )