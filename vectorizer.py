"""
Vectorization Module
Supports: Bag of Words, TF-IDF, and HuggingFace Transformer Tokenization
"""

import numpy as np
import joblib, os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ── 1. Bag of Words ────────────────────────────────────────────────────────────
def get_bow_vectorizer(max_features=10000):
    return CountVectorizer(max_features=max_features, ngram_range=(1,2),
                           min_df=1, max_df=0.95)

# ── 2. TF-IDF ──────────────────────────────────────────────────────────────────
# CHANGE TO THIS:
def get_tfidf_vectorizer(max_features=10000):
    return TfidfVectorizer(max_features=max_features, ngram_range=(1,2),
                           sublinear_tf=True, min_df=1, max_df=0.95)

def fit_vectorizer(texts: list, method: str = "tfidf", save_path: str = None):
    vec = get_tfidf_vectorizer() if method == "tfidf" else get_bow_vectorizer()
    X = vec.fit_transform(texts)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(vec, save_path)
        print(f"✅ Vectorizer saved: {save_path}")
    return vec, X

def load_vectorizer(path: str):
    return joblib.load(path)

def transform(vectorizer, texts: list):
    return vectorizer.transform(texts)

# ── 3. HuggingFace Transformer Tokenizer ──────────────────────────────────────
class TransformerTokenizer:
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name

    def encode(self, texts: list, max_length: int = 128):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def encode_single(self, text: str, max_length: int = 128):
        return self.encode([text], max_length)


if __name__ == "__main__":
    corpus = [
        "This product is amazing and works great!",
        "Terrible quality, do not buy this product.",
        "Average item, nothing special about it.",
        "Great value for money, highly recommend!",
        "Worst purchase ever, broke after one day.",
        "Excellent product, very satisfied with quality.",
    ]
    vec, X = fit_vectorizer(corpus, method="tfidf")
    print("TF-IDF shape:", X.shape)
    print("Vocabulary sample:", list(vec.vocabulary_.keys())[:10])