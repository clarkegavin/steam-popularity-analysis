from .factory import EvaluatorFactory
from . import classification_evaluator
from . import clustering_evaluators

# Register evaluators
EvaluatorFactory.register_evaluator("classification", classification_evaluator.ClassificationEvaluator)
EvaluatorFactory.register_evaluator("clustering", clustering_evaluators.ClusteringEvaluator)

__all__ = [
    "EvaluatorFactory",
    "classification_evaluator",
    "clustering_evaluators",
]
