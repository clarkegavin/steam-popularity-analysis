from .factory import ModelFactory
from . import naive_bayes_model
from .knn_model import KNNClassificationModel
from .naive_bayes_model import NaiveBayesClassificationModel

# Register models
ModelFactory.register_model("naive_bayes", NaiveBayesClassificationModel)
ModelFactory.register_model('knn', KNNClassificationModel)

__all__ = [
    "ModelFactory",
    "naive_bayes_model",
  'knn_model',]
