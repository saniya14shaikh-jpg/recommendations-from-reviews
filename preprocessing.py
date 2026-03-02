"""
Text Preprocessing Engine
Pipeline: raw text → lowercase → URL removal → punctuation removal → 
          stopword removal → normalization → tokenization
"""

import re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
for pkg in ["punkt","stopwords","wordnet","omw-1.4","punkt_tab"]:
    nltk.download(pkg, quiet=True)

_STOP     = set(stopwords.words("english"))
_STEMMER  = PorterStemmer()
_LEMMA    = WordNetLemmatizer()

# Keep negation words — they flip sentiment
KEEP_WORDS = {"not","no","never","nor","neither","none","nothing","without",
              "don't","doesn't","didn't","won't","wouldn't","can't","couldn't"}

def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)

def remove_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)

def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans(string.punctuation, " "*len(string.punctuation)))

def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", " ", text)

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def preprocess(text: str, lemmatize: bool = True, stem: bool = False) -> str:
    """Full preprocessing pipeline — returns cleaned string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = remove_urls(text)
    text = remove_html(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = normalize_whitespace(text)

    tokens = word_tokenize(text)
    tokens = [t for t in tokens
              if t not in _STOP or t in KEEP_WORDS]

    if lemmatize:
        tokens = [_LEMMA.lemmatize(t) for t in tokens]
    elif stem:
        tokens = [_STEMMER.stem(t) for t in tokens]

    return " ".join(tokens)

def preprocess_batch(texts: list, **kwargs) -> list:
    return [preprocess(t, **kwargs) for t in texts]

def get_tokens(text: str) -> list:
    return preprocess(text).split()

if __name__ == "__main__":
    sample = "This product is AMAZING!!! Check https://amazon.com. Not disappointed at all. 10/10"
    print("Original:", sample)
    print("Cleaned: ", preprocess(sample))
    