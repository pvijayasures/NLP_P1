from sklearn.svm import LinearSVC


def build_model(random_state: int | None = None):
    return LinearSVC(
        class_weight="balanced",
        random_state=random_state
    )