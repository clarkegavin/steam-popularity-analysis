from .factory import EvaluatorFactory
from . import classification_evaluator

# Register evaluators
EvaluatorFactory.register_evaluator("classification", classification_evaluator.ClassificationEvaluator)

__all__ = [
    "EvaluatorFactory",
    "classification_evaluator",
]
