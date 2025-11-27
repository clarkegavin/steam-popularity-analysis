from imblearn.over_sampling import RandomOverSampler
from .base import Sampler
from logs.logger import get_logger

class OverSampler(Sampler):
    """
    Wrapper for RandomOverSampler from imblearn.
    """

    def __init__(self, name: str = "over_sampler", **kwargs):
        self.name = name
        self.sampler = RandomOverSampler(**kwargs)
        self.logger = get_logger("OverSampler")
        self.logger.info(f"Initialized OverSampler with name: {self.name} and params: {kwargs}")

    def fit_resample(self, X, y):
        self.logger.info("Starting OverSampler fit_resample")
        return self.sampler.fit_resample(X, y)
