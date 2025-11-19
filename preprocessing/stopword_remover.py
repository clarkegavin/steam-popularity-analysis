# preprocessing/stopword_remover.py
from typing import Iterable, List, Optional, Set, Any
from .base import Preprocessor
from logs.logger import get_logger

# dynamic imports to avoid hard dependency at import time
try:
    import importlib
    _nltk_corpus = importlib.import_module("nltk.corpus")
    _nltk_tokenize = importlib.import_module("nltk.tokenize")
    _nltk_stopwords = getattr(_nltk_corpus, "stopwords", None)
    _word_tokenize = getattr(_nltk_tokenize, "word_tokenize", None)
except Exception:
    _nltk_stopwords = None
    _word_tokenize = None


class StopwordRemover(Preprocessor):
    """Removes stopwords from text.

    Parameters
    ----------
    language: str
        Language for stopword lookup (default: "english").
    stopwords: Optional[Iterable[str]]
        Optional explicit stopword list to use. If provided this takes precedence
        over library-provided stopwords.
    lower: bool
        If True, text is lowercased prior to stopword removal.

    Behaviour
    ---------
    - If NLTK stopwords are available they will be used (unless explicit list
      is provided). If not available and no explicit list provided a small
      default stopword set will be used.
    - Tokenization uses NLTK's word_tokenize when available, otherwise a
      simple split() is used.
    """

    DEFAULT_STOPWORDS: Set[str] = {
        "the",
        "and",
        "is",
        "in",
        "it",
        "of",
        "to",
        "a",
        "for",
        "on",
        "with",
        "that",
        "this",
        "as",
        "are",
        "was",
    }

    def __init__(self, language: str = "english", stopwords: Optional[Iterable[str]] = None, lower: bool = True):
        self.logger = get_logger(self.__class__.__name__)
        self.language = language
        self.lower = bool(lower)
        self.logger.info(f"Initializing StopwordRemover language={language} lower={self.lower}")

        # determine stopword set
        if stopwords is not None:
            try:
                self.stopwords = set(s for s in stopwords if s is not None)
                self.logger.info("Using explicit stopword list provided in params")
            except Exception:
                self.logger.warning("Provided stopwords not iterable; falling back to defaults")
                self.stopwords = set(self.DEFAULT_STOPWORDS)
        else:
            # try to load from nltk if available
            if _nltk_stopwords is not None:
                try:
                    self.stopwords = set(_nltk_stopwords.words(self.language))
                    self.logger.info(f"Loaded {len(self.stopwords)} stopwords for language '{self.language}' from NLTK")
                except Exception:
                    self.logger.warning(f"NLTK stopwords for '{self.language}' not available; using default set")
                    self.stopwords = set(self.DEFAULT_STOPWORDS)
            else:
                self.logger.warning("NLTK stopwords not available; using small built-in default set")
                self.stopwords = set(self.DEFAULT_STOPWORDS)

        # choose tokenizer
        self._tokenize = _word_tokenize if _word_tokenize is not None else None

    def fit(self, X: Iterable[str]):
        # stateless
        return self

    def transform(self, X: Iterable[str]) -> List[str]:
        self.logger.info("Applying StopwordRemover transformation")
        out: List[str] = []
        for i, doc in enumerate(X):
            s = "" if doc is None else str(doc)
            if self.lower:
                try:
                    s = s.lower()
                except Exception:
                    pass

            # tokenize
            try:
                tokens = self._tokenize(s) if self._tokenize is not None else s.split()
            except Exception:
                tokens = s.split()

            # filter stopwords
            try:
                filtered = [t for t in tokens if t not in self.stopwords]
            except Exception:
                # in case tokens are not hashable etc.
                filtered = [t for t in tokens if t not in set(self.stopwords)]

            out_doc = " ".join(filtered)
            out.append(out_doc)

            if i < 3:
                # log small samples
                try:
                    self.logger.info(f"Original: {s.encode('utf-8', errors='ignore')}")
                    self.logger.info(f"Filtered: {out_doc.encode('utf-8', errors='ignore')}")
                except Exception:
                    pass

        self.logger.info("Completed StopwordRemover transformation")
        return out

    def get_params(self) -> dict:
        return {"language": self.language, "lower": self.lower, "stopwords_count": len(self.stopwords)}

