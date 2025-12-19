#pipelines/feature_scaler_pipeline.py
from scalers.factory import ScalerFactory
from logs.logger import get_logger
from typing import Any, Dict

class FeatureScalerPipeline:
    """Pipeline to apply feature scaling using a specified scaler."""

    def __init__(self, scaler_config: Dict[str, Any]):
        self.logger = get_logger("FeatureScalerPipeline")
        self.scaler = ScalerFactory.get_scaler(scaler_config)
        if self.scaler:
            self.logger.info(f"Initialized scaler: {scaler_config.get('name')}")
        else:
            self.logger.info("No scaler configured; proceeding without scaling.")

    def fit(self, X: Any):
        """Fit the scaler to the data."""
        if self.scaler:
            self.logger.info("Fitting scaler to data.")
            self.scaler.fit(X)
        else:
            self.logger.info("No scaler to fit.")

    def transform(self, X: Any) -> Any:
        """Transform the data using the fitted scaler."""
        if self.scaler:
            self.logger.info("Transforming data using scaler.")
            return self.scaler.transform(X)
        else:
            self.logger.info("No scaler to transform data; returning original data.")
            return X

    def fit_transform(self, X: Any) -> Any:
        """Fit the scaler and transform the data."""
        if self.scaler:
            self.logger.info("Fitting and transforming data using scaler.")
            return self.scaler.fit_transform(X)
        else:
            self.logger.info("No scaler to fit/transform; returning original data.")
            return X

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FeatureScalerPipeline":
        """Create a FeatureScalerPipeline from a configuration dictionary.

        Example config:
        {
            "scaler": {
                "name": "standard",
                "params": {"with_mean": True, "with_std": True}
            }
        }
        """
        scaler_config = config.get("scaler", {})
        return cls(scaler_config)