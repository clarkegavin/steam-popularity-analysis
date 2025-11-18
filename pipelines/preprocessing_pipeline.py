# pipelines/preprocessing_pipeline.py
from typing import Any, Dict, List, Optional
import pandas as pd
from pipelines.base import Pipeline
from preprocessing.factory import PreprocessorFactory
from logs.logger import get_logger


class PreprocessingPipeline(Pipeline):
    """Pipeline that applies a sequence of text preprocessors to specified columns.

    Config sample:
    preprocessing:
      - name: stemmer
        params: {language: english}
        columns: ['Description']
      - name: lemmatizer
        columns: ['Title']
    """

    def __init__(self, preprocessors, text_field):
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initializing PreprocessingPipeline with preprocessors:")

        self.preprocessors = preprocessors or []
        self.text_field = text_field
        self.logger = get_logger(self.__class__.__name__)

    def execute(self, X):
        self.logger.info(f"Running text preprocessing on field: {self.text_field}")

        texts = X[self.text_field].fillna("").tolist()

        for pre_cfg in self.preprocessors:
            name = pre_cfg["name"]
            params = pre_cfg.get("params", {})
            self.logger.info(f"Applying preprocessor: {name}")
            pre = PreprocessorFactory.create(name, **params)
            texts = pre.fit_transform(texts)

        X = X.copy()
        X[self.text_field] = texts
        return X

    # def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
    #     self.fit(df)
    #     return self.transform(df)
    #
    # def execute(self, data: Optional[pd.DataFrame] = None) -> Any:
    #     if data is None or not isinstance(data, pd.DataFrame):
    #         raise ValueError("Input must be a DataFrame for PreprocessingPipeline")
    #     self.logger.info(f"Running preprocessing pipeline on columns: {[(cols) for _, cols in self._built]}")
    #     return self.fit_transform(data)
