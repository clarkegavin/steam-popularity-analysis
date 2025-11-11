#evaluators/base.py
from abc import ABC, abstractmethod
from logs.logger import get_logger

class Evaluator(ABC):
    """
    Abstract base class for model evaluators.
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.logger = get_logger(f"Evaluator:{name}")
        # store any additional common kwargs if needed
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger.info(f"Initialized base evaluator '{name}' with kwargs: {kwargs}")


    @abstractmethod
    def evaluate(self, y_true, y_pred, prefix: str ='') -> dict:
        """
        Evaluate the model predictions against true values.
        """
        pass