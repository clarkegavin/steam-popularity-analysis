# preprocessing/sequential.py
class SequentialPreprocessor:
    """
    Simple wrapper to apply a list of preprocessors sequentially.
    Each preprocessor must implement fit_transform and transform.
    """
    def __init__(self, steps):
        self.steps = steps

    def fit_transform(self, X):
        for step in self.steps:
            X = step.fit_transform(X)
        return X

    def transform(self, X):
        for step in self.steps:
            X = step.transform(X)
        return X
