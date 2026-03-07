from sklearn.ensemble import RandomForestClassifier


def build_model(random_state: int | None = None):
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )