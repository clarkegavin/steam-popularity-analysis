# preprocessing/__init__.py
# Expose preprocessing interfaces and default implementations
from .base import Preprocessor
from .factory import PreprocessorFactory
from .stemmer import Stemmer
from .lemmatizer import Lemmatizer
from .lowercase import Lowercase
from .stopword_remover import StopwordRemover
from .emoji_remover import EmojiRemover
from .filter_rows import FilterRows

# Register built-in preprocessors
PreprocessorFactory.register("stem", Stemmer)
PreprocessorFactory.register("stemmer", Stemmer)
PreprocessorFactory.register("lemmatize", Lemmatizer)
PreprocessorFactory.register("lemmatizer", Lemmatizer)
PreprocessorFactory.register("lowercase", Lowercase)
PreprocessorFactory.register("lower", Lowercase)
PreprocessorFactory.register("stopword_remover", StopwordRemover)
PreprocessorFactory.register("stopwords", StopwordRemover)
PreprocessorFactory.register("emoji_remover", EmojiRemover)
PreprocessorFactory.register("emoji", EmojiRemover)
PreprocessorFactory.register("filter_rows", FilterRows)
PreprocessorFactory.register("filter", FilterRows)

__all__ = ["Preprocessor", "PreprocessorFactory", "Stemmer", "Lemmatizer", "Lowercase", "StopwordRemover", "EmojiRemover", "FilterRows"]
