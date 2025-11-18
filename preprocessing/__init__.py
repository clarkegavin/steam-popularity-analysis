# preprocessing/__init__.py
# Expose preprocessing interfaces and default implementations
from .base import Preprocessor
from .factory import PreprocessorFactory
from .stemmer import Stemmer
from .lemmatizer import Lemmatizer

# Register built-in preprocessors
PreprocessorFactory.register("stem", Stemmer)
PreprocessorFactory.register("stemmer", Stemmer)
PreprocessorFactory.register("lemmatize", Lemmatizer)
PreprocessorFactory.register("lemmatizer", Lemmatizer)

__all__ = ["Preprocessor", "PreprocessorFactory", "Stemmer", "Lemmatizer"]

