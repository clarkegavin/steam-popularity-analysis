# models/kmeans_clusterer.py
from sklearn.cluster import KMeans
from .base import Model
from logs.logger import get_logger


class KMeansClusterer(Model):
    """Simple wrapper around sklearn.cluster.KMeans to match project Model API."""

    def __init__(self, name: str = None, **params):
        super().__init__(name, **params)
        self.logger = get_logger(f"KMeansClusterer.{name}")
        self.logger.info(f"Initializing KMeans clusterer with name={name} and params={params}")

    def build(self):
        # sklearn KMeans accepts n_clusters, random_state etc. We pass params directly.
        self.model = KMeans(**self.params)
        self.logger.info(f"Built KMeans model with params={self.params}")
        return self

    def fit(self, X, y=None):
        if self.model is None:
            self.build()
        self.model.fit(X)
        return self

    def fit_predict(self, X, y=None, X_test=None):
        if self.model is None:
            self.build()
        labels = self.model.fit_predict(X)
        return labels

