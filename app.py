"""
Flask REST API Backend
"""

import os, sys, csv, io
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess
from predictor     import predict_text, predict_batch
from database      import save_prediction, get_all_predictions, get_stats

app = Flask(__name__,
            static_folder="../frontend",
            template_folder="../frontend/templates")
CORS(app)

@app.route("/")
def index():
    return send_from_directory("../frontend/templates", "index.html")

@app.route("/api/health")
def health():
    return jsonify({"status":"ok",
                    "message":"Recommendations from Reviews API is running"})

@app.route("/api/predict", methods=["POST"])
def predict():
    data  = request.get_json()
    text  = data.get("text","").strip()
    model = data.get("model","classical")
    if not text:
        return jsonify({"error":"No text provided"}), 400
    result = predict_text(text, model)
    save_prediction(text, result["cleaned_text"], result["sentiment"],
                    result["label"], result["confidence"], model)
    return jsonify(result)

@app.route("/api/predict/batch", methods=["POST"])
def batch_predict():
    data  = request.get_json()
    texts = data.get("texts", [])
    model = data.get("model","classical")
    if not texts or not isinstance(texts, list):
        return jsonify({"error":"Provide a 'texts' list"}), 400
    if len(texts) > 500:
        return jsonify({"error":"Max 500 reviews per batch"}), 400
    results = predict_batch(texts, model)
    for txt, res in zip(texts, results):
        save_prediction(txt, res["cleaned_text"], res["sentiment"],
                        res["label"], res["confidence"], model)
    return jsonify({
        "total":    len(results),
        "positive": sum(1 for r in results if r["label"]==1),
        "negative": sum(1 for r in results if r["label"]==0),
        "results":  results
    })

@app.route("/api/upload", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error":"No file uploaded"}), 400
    f       = request.files["file"]
    model   = request.form.get("model","classical")
    content = f.read().decode("utf-8")
    reader  = csv.DictReader(io.StringIO(content))
    texts   = []
    for row in reader:
        for col in ["review_text","text","review","Review","Text"]:
            if col in row and row[col].strip():
                texts.append(row[col].strip())
                break
    if not texts:
        return jsonify({"error":"No 'review_text' column found"}), 400
    texts   = texts[:500]
    results = predict_batch(texts, model)
    for txt, res in zip(texts, results):
        save_prediction(txt, res["cleaned_text"], res["sentiment"],
                        res["label"], res["confidence"], model)
    return jsonify({
        "total":    len(results),
        "positive": sum(1 for r in results if r["label"]==1),
        "negative": sum(1 for r in results if r["label"]==0),
        "results":  results
    })

@app.route("/api/history")
def history():
    limit = int(request.args.get("limit",100))
    return jsonify(get_all_predictions(limit))

@app.route("/api/stats")
def stats():
    return jsonify(get_stats())

@app.route("/api/preprocess", methods=["POST"])
def preprocess_endpoint():
    data = request.get_json()
    text = data.get("text","")
    return jsonify({
        "original": text,
        "cleaned":  preprocess(text),
        "tokens":   preprocess(text).split()
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)