# imputers/simple_imputer.py
from .base import Imputer
from logs.logger import get_logger
import pandas as pd
from typing import Dict, Any, Optional, List


class SimpleImputer(Imputer):
    """Simple imputer that fills missing values per-column.

    Behavior:
    - For numeric columns: fill missing values with column mean (computed on fit or computed on the fly)
    - For object/string columns: fill missing values with empty string ''

    Parameters
    - columns: optional list of columns to target; if omitted, operates on all DataFrame columns
    - numeric_strategy: optional, currently 'mean' and 'zero' supported
    """

    def __init__(self, columns: Optional[List[str]] = None, numeric_strategy: str = 'mean', text_strategy: str = ''):
        """Create a SimpleImputer.

        Parameters:
        - columns: list of column names to impute (None => all columns)
        - numeric_strategy: 'mean' or 'zero'
        - replace_with: string to use when imputing text/missing non-numeric values
        """
        self.columns = columns
        self.numeric_strategy = numeric_strategy
        self.replace_with = text_strategy
        self.statistics_: Dict[str, Any] = {}
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"Initialized SimpleImputer(columns={self.columns}, numeric_strategy={self.numeric_strategy}, replace_with={self.replace_with})")

    def fit(self, X: pd.DataFrame):
        self.logger.info("Imputer - Starting fit")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("SimpleImputer.fit expects a pandas DataFrame")

        cols = self.columns or list(X.columns)
        for c in cols:
            if c not in X.columns:
                self.logger.warning(f"Column '{c}' not found in DataFrame during fit; skipping")
                continue

            ser = X[c]
            if pd.api.types.is_numeric_dtype(ser):
                if self.numeric_strategy == 'mean':
                    mean_val = ser.dropna().astype(float).mean()
                    self.statistics_[c] = mean_val
                elif self.numeric_strategy == 'zero':
                    self.statistics_[c] = 0
                else:
                    self.logger.warning(f"Unknown numeric_strategy '{self.numeric_strategy}' - skipping numeric stat for {c}")
            else:
                # For non-numeric we will use configured replace_with (default '')
                self.statistics_[c] = self.replace_with

        self.logger.info(f"Computed statistics for imputation: {self.statistics_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Imputer - Starting transform")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("SimpleImputer.transform expects a pandas DataFrame")

        df = X.copy()
        cols = self.columns or list(df.columns)
        for c in cols:
            if c not in df.columns:
                self.logger.debug(f"Column {c} missing from DataFrame; skipping")
                continue

            stat = self.statistics_.get(c, None)
            if stat is None:
                # if numeric and no statistic computed, compute on the fly
                ser = df[c]
                if pd.api.types.is_numeric_dtype(ser):
                    stat = ser.dropna().astype(float).mean()
                else:
                    stat = self.replace_with

            try:
                df[c] = df[c].fillna(stat)
            except Exception as e:
                self.logger.warning(f"Failed to fill column {c} with {stat}: {e}")

        return df

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Imputer - Starting fit_transform")
        return self.fit(X).transform(X)

    def get_params(self):
        return {'columns': self.columns, 'numeric_strategy': self.numeric_strategy, 'replace_with': self.replace_with}
