# pipelines/feature_encoder_pipeline.py
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from logs.logger import get_logger
from pipelines.base import Pipeline
from encoders.factory import EncoderFactory

class FeatureEncoderPipeline(Pipeline):
    """Encodes features (X columns) using multiple encoders from configuration."""

    def __init__(self, encoders: List[Tuple[Any, List[str]]]):
        """
        encoders: List of (encoder_instance, [columns]) pairs.
        encoder_instance must have fit/transform methods.
        """
        self._encoders = encoders
        self.logger = get_logger(self.__class__.__name__)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "FeatureEncoderPipeline":
        raw_entries = cfg.get("encoders") or [cfg.get("encoder")]
        encoders: List[Tuple[Any, List[str]]] = []

        for entry in raw_entries:
            if not entry:
                continue
            name = entry.get("name")
            cols = entry.get("columns") or entry.get("cols")
            params = entry.get("params", {})

            if isinstance(cols, str):
                cols = [cols]

            for col in cols:
                encoder_inst = EncoderFactory.create(name, **params)
                encoders.append((encoder_inst, [col]))

        return cls(encoders)

    def fit(self, df: pd.DataFrame) -> "FeatureEncoderPipeline":
        for encoder, cols in self._encoders:
            for col in cols:
                if col in df.columns:
                    encoder.fit(df[col])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for encoder, cols in self._encoders:
            for col in cols:
                if col in out.columns:
                    out[col] = encoder.transform(out[col])
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a DataFrame for FeatureEncoderPipeline")
        self.logger.info(f"Encoding features: {[col for _, cols in self._encoders for col in cols]}")
        return self.fit_transform(df)
