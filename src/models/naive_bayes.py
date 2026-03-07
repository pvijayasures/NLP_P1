from sklearn.naive_bayes import MultinomialNB


def build_model(random_state: int | None = None):
    return MultinomialNB()