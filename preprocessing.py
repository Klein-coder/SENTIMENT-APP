

from typing import Optional, Set, Iterable, List

import nltk
for resource in ["punkt", "punkt_tab", "wordnet", "omw-1.4", "stopwords"]:
    nltk.download(resource)
import re
from sklearn.base import BaseEstimator, TransformerMixin


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# -------------------------
# LemmaTokenizer (exact)
# -------------------------
class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()

    def __call__(self, reviews):
 
        return [self.wordnetlemma.lemmatize(word) for word in word_tokenize(reviews)]


# Stopwords setup (matching training)

_stop_words = stopwords.words('english')
_new_stopwords = ["movie", "one", "film", "would", "shall", "could", "might"]
_stop_words.extend(_new_stopwords)

if "not" in _stop_words:
    _stop_words.remove("not")
STOP_WORDS = set(_stop_words)


# Helper functions

def contraction_expansion(content: str) -> str:
    if content is None:
        return ""
    s = str(content)
    s = re.sub(r"won\'t", "would not", s)
    s = re.sub(r"can\'t", "can not", s)
    s = re.sub(r"don\'t", "do not", s)
    s = re.sub(r"shouldn\'t", "should not", s)
    s = re.sub(r"needn\'t", "need not", s)
    s = re.sub(r"hasn\'t", "has not", s)
    s = re.sub(r"haven\'t", "have not", s)
    s = re.sub(r"weren\'t", "were not", s)
    s = re.sub(r"mightn\'t", "might not", s)
    s = re.sub(r"didn\'t", "did not", s)
    s = re.sub(r"n\'t", " not", s)  
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\'s", " is", s)
    s = re.sub(r"\'d", " would", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r"\'m", " am", s)
    s = re.sub(r"\'ve", " have", s)
    return s

def remove_url(content: str) -> str:
    if content is None:
        return ""
    return re.sub(r'http\S+', '', str(content))

def remove_special_character(content: str) -> str:
    if content is None:
        return ""
    
    return re.sub(r'[^a-zA-Z0-9\s]', '', str(content))

def remove_stopwords_text(content: str, stop_words_set=STOP_WORDS) -> str:
    if content is None:
        return ""
    clean_data = []
    for token in str(content).split():
        tok = token.strip().lower()
        if tok and tok.isalpha() and tok not in stop_words_set:
            clean_data.append(tok)
    return " ".join(clean_data)

# TextCleaner transformer (exact)

from typing import Optional, Set, Iterable, List
from sklearn.base import BaseEstimator, TransformerMixin

class TextCleaner(BaseEstimator, TransformerMixin):
    
    def __init__(self, stop_words_set: Optional[Set[str]] = None):
        if stop_words_set is None:
            self.stop_words_set: Set[str] = set(STOP_WORDS)  # copy global STOP_WORDS
        else:
            self.stop_words_set = set(stop_words_set)     
            
            
    def fit(self, X: Iterable[str], y=None):
        # no fitting required for this transformer
        return self

    def _clean_one(self, text: str) -> str:
        if text is None:
            return ""
        s = str(text)

        # 1) contraction expansion (do this BEFORE removing punctuation)
        s = contraction_expansion(s)

        # 2) remove urls
        s = remove_url(s)

        # 3) remove special characters (non-alphanumeric)
        s = remove_special_character(s)

        # 4) lowercase and strip
        s = s.lower().strip()

        # 5) remove stopwords (keeps 'not' because we removed it from the set earlier)
        s = remove_stopwords_text(s, self.stop_words_set)

        return s

    def transform(self, X: Iterable[str]) -> List[str]:
       
        return [self._clean_one(x) for x in X]


# Expose names for import *

__all__ = ["LemmaTokenizer", "TextCleaner", "STOP_WORDS", "contraction_expansion",
           "remove_url", "remove_special_character", "remove_stopwords_text"]
