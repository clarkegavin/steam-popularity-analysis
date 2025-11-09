"""
encoders package initialization
Exposes the factory and commonly used encoders for easy imports:
from encoders import EncoderFactory, SklearnLabelEncoder
"""
from .factory import EncoderFactory
from .label_encoder import SklearnLabelEncoder

# Register
EncoderFactory.register("sklearn_label", SklearnLabelEncoder)
EncoderFactory.register("label", SklearnLabelEncoder)

__all__ = ["EncoderFactory", "SklearnLabelEncoder"]

