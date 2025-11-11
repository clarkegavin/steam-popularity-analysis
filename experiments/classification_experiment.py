# experiments/classification_experiment.py
from experiments.base import Experiment
from models.factory import ModelFactory
from evaluators.factory import EvaluatorFactory
from logs.logger import get_logger
import mlflow
import json, os
from datetime import datetime
from typing import Optional, Dict, List

class ClassificationExperiment(Experiment):
    def __init__(
        self,
        name: str,
        model_name: str,
        evaluator_name: str,
        metrics: List[str],
        save_path: Optional[str] = None,
        model_params: Optional[Dict] = None,
        evaluator_params: Optional[Dict] = None,
        mlflow_tracking: bool = True,
        mlflow_experiment: Optional[str] = None,
    ):

        super().__init__(name, mlflow_tracking, mlflow_experiment)
        self.model_name = model_name
        self.evaluator_name = evaluator_name
        self.metrics = metrics
        self.save_path = save_path
        self.model_params = model_params or {}
        self.evaluator_params = evaluator_params or {}
        self.logger = get_logger(f"ClassificationExperiment:{name}")
        self.logger.info('Initializing classification experiment')
        self.logger.info(f'Classification model name: {model_name}')
        self.logger.info(f'Classification model params: {self.model_params}')
        self.model = ModelFactory.get_model(model_name, **self.model_params)
        self.logger.info(f'Classification model: {self.model}')
        self.logger.info(f'Evaluator params: {self.evaluator_params}')
        self.evaluator = EvaluatorFactory.get_evaluator(name=evaluator_name,  **self.evaluator_params)
        self.results = {}

    def run(self, X_train, X_test, y_train, y_test):
        self.logger.info(f"Running classification experiment '{self.name}'")

        with mlflow.start_run(run_name=self.name):
            mlflow.log_param("model", self.model_name)
            mlflow.log_param("evaluator", self.evaluator_name)
            mlflow.log_params(self.model_params)

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            self.results = self.evaluator.evaluate(y_test, y_pred)

            for metric_name, value in self.results.items():
                mlflow.log_metric(metric_name, value)

            try:
                mlflow.sklearn.log_model(self.model, artifact_path="model")
            except Exception as e:
                self.logger.warning(f"Could not log model to MLflow: {e}")

            if self.save_path:
                self.save_results()

        self.logger.info(f"Experiment '{self.name}' complete.")
        return self.results

    def save_results(self):
        os.makedirs(self.save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.save_path, f"{self.name}_{timestamp}.json")
        with open(file_path, "w") as f:
            json.dump(
                {
                    "experiments": self.name,
                    "model": self.model_name,
                    "metrics": self.results,
                    "params": self.model_params,
                    "timestamp": timestamp,
                },
                f,
                indent=4,
            )
        self.logger.info(f"Saved results locally to {file_path}")
