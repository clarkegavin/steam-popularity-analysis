# reducers/__init__.py
from .base import Reducer
from .umap_reducer import UMAPReducer
from .factory import ReducerFactory

# register default reducers
ReducerFactory.register('umap', UMAPReducer)

__all__ = [
    "Reducer",
    "UMAPReducer",
    "ReducerFactory",
]

