# pipelines/experiment_pipeline.py
from typing import Dict, Any, Optional, List
from .base import Pipeline
from logs.logger import get_logger
from experiments.factory import ExperimentFactory

class ExperimentPipeline(Pipeline):
    """
    Pipeline that runs one or more experiments for a given model type.
    """
    def __init__(
        self,
        experiment_type: str,
        model_name: str,
        evaluator_name: str,
        metrics: List[str],
        experiments: Optional[List[Dict[str, Any]]] = None,
        mlflow_experiment: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name or model_name)
        self.experiment_type = experiment_type
        self.model_name = model_name
        self.evaluator_name = evaluator_name
        self.metrics = metrics
        self.experiments = experiments or [{}]
        self.mlflow_experiment = mlflow_experiment
        self.logger = get_logger(self.__class__.__name__)

    @classmethod
    def from_config(cls, entry: Dict[str, Any]) -> "ExperimentPipeline":
        params = entry.get("params", {})
        return cls(**params, name=entry.get("name"))

    def execute(self, X_train, X_test, y_train, y_test):
        self.logger.info(f"Running experiments for model '{self.model_name}'")
        mlflow_experiment_name = self.mlflow_experiment or f"{self.model_name}_experiments"

        for i, exp_cfg in enumerate(self.experiments, start=1):
            run_name = exp_cfg.get("run_name", f"{self.model_name}_run{i}")
            self.logger.info(f"Starting experiment {i} ({run_name}) with params {exp_cfg.get('params', {})}")

            exp_params = {
                "name": run_name,
                "model_name": self.model_name,
                "evaluator_name": self.evaluator_name,
                "metrics": self.metrics,
                "mlflow_experiment": mlflow_experiment_name,
                **exp_cfg.get("params", {})
            }

            self.logger.info(f'Experiment parameters: {exp_params}')
            experiment = ExperimentFactory.get_experiment(self.experiment_type, **exp_params)
            self.logger.info(f"Created experiment instance: {experiment}")
            if experiment:
                self.logger.info(f"Executing experiment '{run_name}'")
                experiment.run(X_train, X_test, y_train, y_test)

        self.logger.info(f"All experiments for '{self.model_name}' complete.")
