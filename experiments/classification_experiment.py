# experiments/classification_experiment.py
from experiments.base import Experiment
from models.factory import ModelFactory
from evaluators.factory import EvaluatorFactory
from logs.logger import get_logger
import mlflow
import json, os
from datetime import datetime
from typing import Optional, Dict, List, Any
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from vectorizers.factory import VectorizerFactory
from mlflow.models import infer_signature
from visualisations.factory import VisualisationFactory


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
        description: Optional[str] = None,
        target_encoder: Optional[Any] = None,
        vectorizer: Optional[Dict] = None,
        visualisations: Optional[List[Dict]] = None,
        **kwargs
    ):

        super().__init__(name, mlflow_tracking, mlflow_experiment)

        # Logger
        self.logger = get_logger(f"ClassificationExperiment:{name}")
        self.logger.info("Initializing classification experiment")

        # Standard fields
        self.name = name
        self.model_name = model_name
        self.metrics = metrics
        self.description = description
        self.evaluator_name = evaluator_name
        self.save_path = save_path
        self.target_encoder = target_encoder

        # --- FLEXIBLE PARAMETER HANDLING -------------------------
        # Everything not explicitly defined is inside kwargs
        # Example: n_neighbors, weights, cv_enabled, vectorizer override etc.
        self.extra_params = kwargs


        # Vectorizer
        self.vectorizer = vectorizer or {}
        self.vectorizer_name = self.vectorizer.get("vectorizer_name")
        self.vectorizer_field = self.vectorizer.get("vectorizer_field")
        self.vectorizer_params = self.vectorizer.get("vectorizer_params", {})

        self.visualisations = visualisations or []

        # Model parameters: merge YAML model_params + extra params that belong to the model
        self.model_params = (model_params or {}).copy()
        for k, v in kwargs.items():
            if k not in ["cv_enabled", "cv_folds", "cv_shuffle", "cv_random_state",
                         "cv_stratified", "visualisations", "vectorizer"]:
                self.model_params.setdefault(k, v)

        self.logger.info(f"Model params resolved: {self.model_params}")

        # Cross-validation params (default off unless YAML enables)
        self.cv_enabled = kwargs.get("cv_enabled", False)
        self.cv_folds = kwargs.get("cv_folds", 5)
        self.cv_stratified = kwargs.get("cv_stratified", True)
        self.cv_shuffle = kwargs.get("cv_shuffle", True)
        self.cv_random_state = kwargs.get("cv_random_state", 42)

        # Initializing model/evaluator
        self.model = ModelFactory.get_model(self.model_name, **self.model_params)
        self.evaluator = EvaluatorFactory.get_evaluator(name=evaluator_name, **(evaluator_params or {}))

        # Results store
        self.results = {}
        self.logger.info("ClassificationExperiment initialised successfully.")

    def run(self, X_train, X_test, y_train, y_test):
        self.logger.info(f"Running classification experiment '{self.name}'")

               # --- 1. Vectorizer support -----------------------


        if self.vectorizer_name:
            self.logger.info(f"Using vectorizer '{self.vectorizer_name}' on field '{self.vectorizer_field}'")
            self.vectorizer_params['column'] = self.vectorizer_field
            vectorizer = VectorizerFactory.get_vectorizer(
                self.vectorizer_name,
                **self.vectorizer_params
            )
            X_train = vectorizer.fit_transform(X_train.fillna("")) # Fit on training set
            X_test = vectorizer.transform(X_test.fillna("")) # Transform on test set


        with mlflow.start_run(run_name=self.name):
            mlflow.log_param("model", self.model_name)
            mlflow.log_param("evaluator", self.evaluator_name)
            mlflow.log_param("description", self.description)
            mlflow.log_params(self.model_params)

            # --- 2. Cross-validation support -----------------------
            mlflow.log_param("cv_enabled", self.cv_enabled)
            mlflow.log_param("cv_folds", self.cv_folds)
            mlflow.log_param("cv_stratified", self.cv_stratified)

            # preprocessing support
            # preprocessing_params = self.extra_params
            # if hasattr(self.extra_params.get("preprocessing", {}), "get_params"):
            #     self.logger.info("Logging preprocessing parameters to MLflow")
            #     preprocessing_params = self.extra_params["preprocessing"].get_params()
            #     for k, v in preprocessing_params.items():
            #         mlflow.log_param(f"preprocessing_{k}", v)

            if self.cv_enabled:
                self.results = self._run_cross_validation(X_train, y_train)

                # --- Train final model on full training set ---
                self.logger.info("Training final model on full training set")
                final_model = ModelFactory.get_model(self.model_name, **self.model_params)
                final_model.fit(X_train, y_train)

                # Infer signature for logging
                signature = infer_signature(X_train, final_model.predict(X_train))

                # Log final model
                try:
                    mlflow.sklearn.log_model(final_model,
                                             name="model",
                                             signature=signature,
                                             registered_model_name=self.model_name)
                    self.logger.info("Final model logged to MLflow successfully.")
                except Exception as e:
                    self.logger.warning(f"Could not log model to MLflow: {e}")

                # --- Evaluate on test set if provided ---
                if X_test is not None and y_test is not None:
                    test_metrics = self._run_test_evaluation(final_model, X_test, y_test)
                    self.results.update(test_metrics)

                # Save results locally
                if self.save_path:
                    self.save_results()

            return self.results


    def _run_test_evaluation(self, model, X_test, y_test):
        self.logger.info("Running test set evaluation")
        y_pred = model.predict(X_test)
        test_metrics = self.evaluator.evaluate(y_test, y_pred)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        self.logger.info(f"Test set metrics: {test_metrics}")

        # Generate and log visualisations
        self.logger.info("Generating test set visualisations")
        self._generate_visualisations(y_test, y_pred)

        return test_metrics


    def _run_cross_validation(self, X, y):
        self.logger.info(f"Running {self.cv_folds}-fold cross-validation "
                         f"(stratified={self.cv_stratified})")

        # Create folds
        if self.cv_stratified:
            self.logger.info(f"Using StratifiedKFold for cross-validation with {self.cv_folds} folds, shuffle={self.cv_shuffle}, and random_state={self.cv_random_state}")
            splitter = StratifiedKFold(n_splits=self.cv_folds, shuffle=self.cv_shuffle, random_state=self.cv_random_state)
        else:
            self.logger.info(
                f"Using KFold for cross-validation with {self.cv_folds} folds, shuffle={self.cv_shuffle}, and random_state={self.cv_random_state}")
            splitter = KFold(n_splits=self.cv_folds, shuffle=self.cv_shuffle, random_state=self.cv_random_state)

        fold_metrics = {metric: [] for metric in self.metrics}
        fold_index = 1
        X = X.to_numpy() if hasattr(X, "to_numpy") else X
        y = y.to_numpy() if hasattr(y, "to_numpy") else y

        for train_idx, val_idx in splitter.split(X, y):
            self.logger.info(f"Fold {fold_index}/{self.cv_folds}")

            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Recreate a fresh model each fold
            model = ModelFactory.get_model(self.model_name, **self.model_params)
            model.fit(X_train_fold, y_train_fold)

            y_pred = model.predict(X_val_fold)
            current_res = self.evaluator.evaluate(y_val_fold, y_pred)

            # Store metrics
            for m in self.metrics:
                fold_metrics[m].append(current_res[m])
                mlflow.log_metric(f"fold_{fold_index}_{m}", current_res[m])

            fold_index += 1
            # Generate and log visualisations
            self.logger.info(f"Generating visualisations for fold {fold_index - 1}")
            self._generate_visualisations(y_val_fold, y_pred, fold=fold_index - 1)

        # Compute average CV metrics
        averaged_metrics = {m: float(sum(vals) / len(vals)) for m, vals in fold_metrics.items()}
        std_metrics = {f"{m}_std": float(np.std(vals)) for m, vals in fold_metrics.items()}

        # Log averaged metrics to MLflow
        for m, avg in averaged_metrics.items():
            mlflow.log_metric(f"cv_mean_{m}", avg)
        # log std metrics to MLflow
        for m_std, std in std_metrics.items():
            mlflow.log_metric(f'cv_{m_std}', std)


        self.results = {**averaged_metrics, **std_metrics}
        self.logger.info(f"Cross-validation results: {self.results}")

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
                }, f, indent=4,
            )
        self.logger.info(f"Saved results locally to {file_path}")

    def _generate_visualisations(self, y_true, y_pred, fold=0):

        if not self.visualisations:
            self.logger.info("No visualisations configured, skipping.")
            return

        self.logger.info("Generating visualisations")
        self.logger.info(f'Target encoder available: {self.target_encoder is not None}')
        for viz_cfg in self.visualisations:
            title_suffix = f"_fold_{fold}" if fold > 0 else ""

            viz_name = viz_cfg.get("name")
            self.logger.info(f'Visualisation to create: {viz_name}')
            viz_kwargs = viz_cfg.get("kwargs", {})
            viz_kwargs.update({
                "y_true": y_true,
                "y_pred": y_pred,
                "target_encoder": self.target_encoder,
            })
            try:
                viz = VisualisationFactory.get_visualisation(viz_name, **viz_kwargs)
                if viz:
                    self.logger.info(f"Creating visualisation: {viz_name}")
                    fig = viz.plot(None)  # 'data' param not needed here
                    # Save locally
                    if self.save_path:
                        os.makedirs(self.save_path, exist_ok=True)
                        filepath = os.path.join(self.save_path, f"{self.name}_{viz_name}{title_suffix}.png")
                        viz.save(fig, filepath)
                        self.logger.info(f"Saved visualisation '{viz_name}{title_suffix}' to {filepath}")
                    # Log to MLflow
                        try:
                            mlflow.log_artifact(filepath)
                        except Exception as e:
                            self.logger.warning(f"Could not log visualisation '{viz_name}' to MLflow: {e}")
            except Exception as e:
                self.logger.warning(f"Could not create visualisation '{viz_name}': {e}")
