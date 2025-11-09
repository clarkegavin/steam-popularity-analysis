# orchestrator/pipeline_orchestrator.py
from typing import List, Optional, Dict
from logs.logger import get_logger
import pandas as pd

from pipelines import TargetFeaturePipeline, DataSplitterPipeline


class FeatureEncoderPipeline:
    pass


class PipelineOrchestrator:
    def __init__(self, pipelines: List, max_retries: int = 3, parallel: bool = False):
        self.logger = get_logger(self.__class__.__name__)
        self.pipelines = pipelines
        self.max_retries = max_retries
        self.parallel = parallel

    def run_pipeline(self, pipeline, data: Optional[pd.DataFrame] = None, extra: Optional[Dict] = None):
        """Run a single pipeline with retry logic."""
        attempt = 0
        while attempt < self.max_retries:
            try:
                self.logger.info(f"Running pipeline: {pipeline.__class__.__name__}, attempt {attempt+1}")

                if extra is None:
                    extra = {}

                # Allow TargetFeaturePipeline to get y_train/y_test explicitly
                if isinstance(pipeline, TargetFeaturePipeline):
                    result = pipeline.execute(
                        y=extra["y"],
                        fit=extra.get("fit", True)
                    )
                else:
                    result = pipeline.execute(data)

                self.logger.info(f"Pipeline {pipeline.__class__.__name__} completed successfully")
                return result
            except Exception as e:
                attempt += 1
                self.logger.error(f"Pipeline {pipeline.__class__.__name__} failed on attempt {attempt}: {e}")

        self.logger.error(f"Pipeline {pipeline.__class__.__name__} failed after {self.max_retries} attempts")
        return None

    def run(self, data: Optional[pd.DataFrame] = None, target_column: str = None):
        """Run all pipelines sequentially."""
        self.logger.info("Starting orchestrator run")
        X_train = X_test = y_train = y_test = None
        current_data = data

        for pipeline in self.pipelines:
            if isinstance(pipeline, DataSplitterPipeline):
                # Split data into train/test
                splits = pipeline.execute(current_data)
                X_train, X_test = splits["X_train"], splits["X_test"]
                y_train, y_test = splits["y_train"], splits["y_test"]
            elif isinstance(pipeline, TargetFeaturePipeline):
                # Fit on training target, transform test target
                y_train_encoded = self.run_pipeline(pipeline, extra={"y": y_train, "fit": True})
                y_test_encoded = self.run_pipeline(pipeline, extra={"y": y_test, "fit": False})
                # Replace y_train/y_test for downstream pipelines if needed
                y_train, y_test = y_train_encoded, y_test_encoded
            elif isinstance(pipeline, FeatureEncoderPipeline):
                X_train = self.run_pipeline(pipeline, data=X_train, extra={"fit": True})
                X_test = self.run_pipeline(pipeline, data=X_test, extra={"fit": False})
            else:
                # Other pipelines that operate on the full dataset
                current_data = self.run_pipeline(pipeline, data=current_data)

        self.logger.info("Pipeline orchestration complete")
        return X_train, X_test, y_train, y_test
