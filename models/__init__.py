from .factory import ModelFactory
from . import naive_bayes_model
from .knn_model import KNNClassificationModel
from .naive_bayes_model import NaiveBayesClassificationModel
from .xgboost_model import XGBoostClassificationModel
from .svm_model import SVMModel
from .hdbscan_clusterer import HDBSCANClusterer

# Register models
ModelFactory.register_model("naive_bayes", NaiveBayesClassificationModel)
ModelFactory.register_model('knn', KNNClassificationModel)
ModelFactory.register_model("xgboost", XGBoostClassificationModel)
ModelFactory.register_model("svm", SVMModel)
ModelFactory.register_model("hdbscan", HDBSCANClusterer)

__all__ = [
    "ModelFactory",
    "naive_bayes_model",
    "knn_model",
    "xgboost_model",
    "svm_model",
    "hdbscan_clusterer",
]
