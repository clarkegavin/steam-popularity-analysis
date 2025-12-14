from .factory import ModelFactory
from . import naive_bayes_model
from .knn_model import KNNClassificationModel
from .naive_bayes_model import NaiveBayesClassificationModel
from .xgboost_model import XGBoostClassificationModel
from .svm_model import SVMModel
from .hdbscan_clusterer import HDBSCANClusterer
from .linear_svc_model import LinearSVCModel
from .linear_regression_model import LinearRegressionModel
from .logistic_regression import LogisticRegressionModel
from .random_forest_classifier_model import RandomForestClassifierModel

# Register models
ModelFactory.register_model("naive_bayes", NaiveBayesClassificationModel)
ModelFactory.register_model('knn', KNNClassificationModel)
ModelFactory.register_model("xgboost", XGBoostClassificationModel)
ModelFactory.register_model("svm", SVMModel)
ModelFactory.register_model("hdbscan", HDBSCANClusterer)
ModelFactory.register_model("linear_svc", LinearSVCModel)
ModelFactory.register_model("linear_regression", LinearRegressionModel)
ModelFactory.register_model("logistic_regression", LogisticRegressionModel)
ModelFactory.register_model("random_forest", RandomForestClassifierModel)




__all__ = [
    "ModelFactory",
    "naive_bayes_model",
    "knn_model",
    "xgboost_model",
    "svm_model",
    "hdbscan_clusterer",
    "linear_svc_model",
    "linear_regression_model",
    "logistic_regression",
    "random_forest_classifier_model",
]
