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
            self.logger.info(f"Preprocessor step: {name} with config: {pre_cfg}")
            params = pre_cfg.get("params", {})
            self.logger.info(f"Applying preprocessor: {name}")
            pre = PreprocessorFactory.create(name, **params)
            texts = pre.fit_transform(texts)

            # Check if 'roblox' still exists after stopword_remover
            if name == "stopword_remover":
                self.logger.info("Checking for 'roblox' presence after stopword removal")
                still_present = [t for t in texts if "roblox" in t.lower()]
                self.logger.info(f"'roblox' still present in {len(still_present)} texts after stopword removal")
                # Optionally print some examples
                for sample in still_present[:5]:
                    self.logger.info(f"Sample text: {sample}")

        X = X.copy()
        self.logger.info(f"Completed text preprocessing on field: {self.text_field}")
        safe_texts = " | ".join(
            t.encode("ascii", errors="ignore").decode() for t in texts[:20]
        )
        #self.logger.info("Sample processed texts: %s", safe_texts)
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
