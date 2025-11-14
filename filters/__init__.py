#filter/__init__.py
from .factory import FilterFactory
from .drop_columns_filter import DropColumnsFilter

from . import drop_columns_filter

FilterFactory.register_filter("drop_columns", DropColumnsFilter)

__all__ = [
    "FilterFactory",
    "drop_columns_filter",
]
