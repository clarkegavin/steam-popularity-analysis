from typing import Any, Iterable, List, Optional, Union
import numpy as np
import pandas as pd

from .base import Encoder


class MultiHotEncoder(Encoder):
    """
    Encoder for multi-label fields where each cell may contain multiple categories
    (e.g., comma-separated platforms). Produces a binary indicator matrix with one
    column per known category and a 1 where the category is present in the cell.

    Parameters:
    - sep: separator used when transforming raw string cells (default: ',')
    - dtype: output dtype (default int)
    - handle_unknown: 'ignore' or 'error' when new categories are seen in transform
    """

    def __init__(self, sep: str = ',', dtype: Optional[type] = None, handle_unknown: str = 'ignore'):
        self.sep = sep
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.categories_: List[str] = []
        self._fitted = False

    def _split_cell(self, cell):
        if pd.isna(cell):
            return []
        if isinstance(cell, (list, tuple, set)):
            return [str(x).strip() for x in cell if x is not None and str(x).strip() != '']
        s = str(cell)
        # split and strip
        parts = [p.strip() for p in s.split(self.sep) if p.strip() != '']
        return parts

    def fit(self, y: Iterable[Any]) -> "MultiHotEncoder":
        # collect unique category tokens across all cells, preserving order
        seen = []
        for cell in y:
            tokens = self._split_cell(cell)
            for t in tokens:
                if t not in seen:
                    seen.append(t)
        self.categories_ = seen
        self._fitted = True
        return self

    def transform(self, y: Iterable[Any]) -> Union[pd.DataFrame, np.ndarray]:
        if not self._fitted:
            raise ValueError("MultiHotEncoder has not been fitted yet. Call fit() first.")

        # Accept Series for index preservation
        index = None
        name = None
        if isinstance(y, pd.Series):
            index = y.index
            name = y.name or 'feature'

        # Prepare output array
        values = list(y)
        n = len(values)
        m = len(self.categories_)
        arr = np.zeros((n, m), dtype=self.dtype if self.dtype is not None else int)
        cat_to_idx = {c: i for i, c in enumerate(self.categories_)}

        for i, cell in enumerate(values):
            tokens = self._split_cell(cell)
            for t in tokens:
                if t not in cat_to_idx:
                    if self.handle_unknown == 'error':
                        raise ValueError(f"Unknown token '{t}' encountered in transform and handle_unknown='error'")
                    else:
                        continue
                arr[i, cat_to_idx[t]] = 1

        if index is not None:
            colnames = [f"{name}__{c}" for c in self.categories_]
            df = pd.DataFrame(arr, index=index, columns=colnames)
            if self.dtype is not None:
                df = df.astype(self.dtype)
            return df
        return arr

    def fit_transform(self, y: Iterable[Any]) -> Union[pd.DataFrame, np.ndarray]:
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y_enc: Iterable[Any]) -> Union[List[List[str]], pd.Series]:
        if not self._fitted:
            raise ValueError("MultiHotEncoder has not been fitted yet. Call fit() first.")

        if isinstance(y_enc, pd.DataFrame):
            arr = y_enc.values
            idx = y_enc.index
        else:
            arr = np.asarray(y_enc)
            idx = None

        out = []
        for row in arr:
            ones = np.where(row != 0)[0]
            tokens = [self.categories_[int(i)] for i in ones]
            out.append(tokens)

        if idx is not None:
            return pd.Series(out, index=idx)
        return out

# alias
MultiHot = MultiHotEncoder

