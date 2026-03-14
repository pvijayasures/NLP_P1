from sklearn.linear_model import LogisticRegression


def build_model(random_state: int | None = None):
    return LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver='lbfgs',
        tol=0.0001,
        class_weight="balanced",
        random_state=random_state
    )