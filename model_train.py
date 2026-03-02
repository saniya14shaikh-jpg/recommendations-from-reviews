"""
Model Training Script
Trains: Logistic Regression, SVM, LSTM (PyTorch)
Usage: python backend/model_train.py
"""

import os, sys, joblib, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_batch
from vectorizer    import fit_vectorizer

DATA_PATH = "amazon_reviews.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path=DATA_PATH):
    print(f"📂 Loading data from {path}...")
    df = pd.read_csv(path)
    df = df.dropna(subset=["review_text","sentiment"])
    df["cleaned"] = preprocess_batch(df["review_text"].tolist())
    print(f"   {len(df)} reviews loaded | "
          f"Positive: {df.sentiment.sum()} | "
          f"Negative: {len(df)-df.sentiment.sum()}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 2. Classical ML
# ─────────────────────────────────────────────────────────────────────────────
def train_classical(df):
    print("\n🔧 Training Classical ML Models (LR + SVM)...")
    X, y = df["cleaned"].tolist(), df["sentiment"].tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    vec, X_tr_vec = fit_vectorizer(
        X_tr, method="tfidf",
        save_path=f"{MODEL_DIR}/tfidf_vectorizer.pkl")
    X_te_vec = vec.transform(X_te)

    results = {}
    for name, clf in [
        ("LogisticRegression", LogisticRegression(max_iter=500, C=1.0)),
        ("LinearSVC",          LinearSVC(max_iter=1000, C=1.0))]:
        clf.fit(X_tr_vec, y_tr)
        preds = clf.predict(X_te_vec)
        acc   = accuracy_score(y_te, preds)
        print(f"   {name} Accuracy: {acc:.4f}")
        joblib.dump(clf, f"{MODEL_DIR}/{name.lower()}_model.pkl")
        results[name] = {"accuracy": acc}

    best = max(results, key=lambda k: results[k]["accuracy"])
    import shutil
    shutil.copy(f"{MODEL_DIR}/{best.lower()}_model.pkl",
                f"{MODEL_DIR}/best_classical_model.pkl")
    print(f"   ✅ Best model: {best} ({results[best]['accuracy']:.4f})")

    with open(f"{MODEL_DIR}/classical_results.json","w") as f:
        json.dump(results, f, indent=2)
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 3. LSTM Model
# ─────────────────────────────────────────────────────────────────────────────
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.labels  = labels
        self.max_len = max_len
        self.vocab   = vocab
        self.data    = [self._encode(t) for t in texts]

    def _encode(self, text):
        tokens = text.split()[:self.max_len]
        ids    = [self.vocab.get(t, 1) for t in tokens]
        pad    = self.max_len - len(ids)
        return ids + [0] * pad

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return (torch.tensor(self.data[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.float))

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128,
                 hidden_dim=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size+2, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, (h, _) = self.lstm(emb)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.sigmoid(self.fc(self.dropout(h))).squeeze()

def build_vocab(texts, max_vocab=15000):
    from collections import Counter
    counts = Counter(t for txt in texts for t in txt.split())
    vocab  = {w:i+2 for i,(w,_) in
              enumerate(counts.most_common(max_vocab))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

def train_lstm(df, epochs=5, batch_size=64, lr=1e-3):
    print("\n🔧 Training LSTM Model (PyTorch)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    texts  = df["cleaned"].tolist()
    labels = df["sentiment"].tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2,
        random_state=42, stratify=labels)

    vocab = build_vocab(X_tr)
    joblib.dump(vocab, f"{MODEL_DIR}/lstm_vocab.pkl")

    tr_ds = ReviewDataset(X_tr, y_tr, vocab)
    te_ds = ReviewDataset(X_te, y_te, vocab)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=batch_size)

    model = SentimentLSTM(len(vocab)).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.BCELoss()
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)

    history = []
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0
        for xb, yb in tqdm(tr_dl, desc=f"Epoch {ep}/{epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()

        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for xb, yb in te_dl:
                p = model(xb.to(device)).cpu().numpy()
                preds_all.extend(p)
                labels_all.extend(yb.numpy())
        preds_bin = [1 if p >= 0.5 else 0 for p in preds_all]
        acc      = accuracy_score(labels_all, preds_bin)
        avg_loss = total_loss / len(tr_dl)
        print(f"   Epoch {ep} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")
        history.append({"epoch":ep, "loss":avg_loss, "val_accuracy":acc})

    torch.save(model.state_dict(), f"{MODEL_DIR}/lstm_model.pt")
    joblib.dump({"vocab_size":len(vocab), "embed_dim":128,
                 "hidden_dim":256, "n_layers":2},
                f"{MODEL_DIR}/lstm_config.pkl")
    with open(f"{MODEL_DIR}/lstm_history.json","w") as f:
        json.dump(history, f, indent=2)
    print(f"   ✅ LSTM saved | Final Val Acc: {history[-1]['val_accuracy']:.4f}")
    return history

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print("📊 Generating dataset...")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dataset", "Data/dataset.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.generate_dataset(5000, DATA_PATH)

    df = load_data(DATA_PATH)
    train_classical(df)
    train_lstm(df, epochs=5)
    print("\n🎉 All models trained successfully!")
    print(f"   Models saved in: {MODEL_DIR}/")

