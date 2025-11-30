# reducers/umap_reducer.py
from typing import Any
from logs.logger import get_logger

try:
    import umap
except Exception:
    umap = None

from .base import Reducer


class UMAPReducer(Reducer):
    def __init__(self, name='umap', n_components: int = 2, random_state: int = 42, **kwargs):
        self.logger = get_logger("UMAPReducer")
        self.name = name
        if umap is None:
            self.logger.warning("umap-learn is not installed; UMAPReducer will raise on fit/transform.")
        self.n_components = n_components
        self.random_state = random_state
        self.kwargs = kwargs
        self._model = None

    def fit(self, X: Any):
        if umap is None:
            raise RuntimeError("umap-learn is required for UMAPReducer. Install with 'pip install umap-learn'.")
        self._model = umap.UMAP(n_components=self.n_components, random_state=self.random_state, **self.kwargs)
        self._model.fit(X)
        return self

    def transform(self, X: Any):
        if self._model is None:
            # lazily create the model if fit() wasn't called
            self._model = umap.UMAP(n_components=self.n_components, random_state=self.random_state, **self.kwargs)
        return self._model.transform(X)

    def fit_transform(self, X: Any):
        if umap is None:
            raise RuntimeError("umap-learn is required for UMAPReducer. Install with 'pip install umap-learn'.")
        self._model = umap.UMAP(n_components=self.n_components, random_state=self.random_state, **self.kwargs)
        return self._model.fit_transform(X)

    def set_components(self, n_components: int):
        self.n_components = n_components
        self._model = None  # reset model to force re-creation with new components

