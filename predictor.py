"""
Prediction Engine — loads trained models and runs inference
"""

import os, sys, joblib
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess
from model_train   import SentimentLSTM

MODEL_DIR = "models"

class ClassicalPredictor:
    def __init__(self):
        self.vec   = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.pkl")
        self.model = joblib.load(f"{MODEL_DIR}/best_classical_model.pkl")

    def predict(self, text: str) -> dict:
        cleaned = preprocess(text)
        X = self.vec.transform([cleaned])
        label = int(self.model.predict(X)[0])
        try:
            prob = float(self.model.predict_proba(X)[0][label])
        except AttributeError:
            prob = 0.95 if label == 1 else 0.05
        return {
            "label": label,
            "sentiment": "positive" if label == 1 else "negative",
            "confidence": round(prob, 4),
            "cleaned_text": cleaned
        }

    def predict_batch(self, texts: list) -> list:
        return [self.predict(t) for t in texts]


class LSTMPredictor:
    def __init__(self):
        self.vocab  = joblib.load(f"{MODEL_DIR}/lstm_vocab.pkl")
        cfg         = joblib.load(f"{MODEL_DIR}/lstm_config.pkl")
        self.device = torch.device("cpu")
        self.model  = SentimentLSTM(**cfg)
        self.model.load_state_dict(
            torch.load(f"{MODEL_DIR}/lstm_model.pt",
                       map_location=self.device))
        self.model.eval()
        self.MAX_LEN = 100

    def _encode(self, text: str):
        tokens = preprocess(text).split()[:self.MAX_LEN]
        ids    = [self.vocab.get(t, 1) for t in tokens]
        ids   += [0] * (self.MAX_LEN - len(ids))
        return torch.tensor([ids], dtype=torch.long)

    def predict(self, text: str) -> dict:
        with torch.no_grad():
            prob = float(self.model(self._encode(text)))
        label = 1 if prob >= 0.5 else 0
        return {
            "label": label,
            "sentiment": "positive" if label == 1 else "negative",
            "confidence": round(prob if label == 1 else 1 - prob, 4),
            "raw_score": round(prob, 4),
            "cleaned_text": preprocess(text)
        }

    def predict_batch(self, texts: list) -> list:
        return [self.predict(t) for t in texts]


# ── Singleton loader ───────────────────────────────────────────────────────────
_classical = None
_lstm      = None

def get_classical():
    global _classical
    if _classical is None:
        _classical = ClassicalPredictor()
    return _classical

def get_lstm():
    global _lstm
    if _lstm is None:
        _lstm = LSTMPredictor()
    return _lstm

def predict_text(text: str, model: str = "classical") -> dict:
    predictor = get_lstm() if model == "lstm" else get_classical()
    return predictor.predict(text)

def predict_batch(texts: list, model: str = "classical") -> list:
    predictor = get_lstm() if model == "lstm" else get_classical()
    return predictor.predict_batch(texts)