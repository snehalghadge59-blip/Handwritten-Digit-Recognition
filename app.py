"""
app.py  –  Flask backend for Handwritten Digit Recognition
"""

import os
import pickle
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps, ImageFilter
import io
from scipy.ndimage import zoom

app = Flask(__name__)

# ── Load model & scaler ──────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
with open(os.path.join(BASE, "model", "knn_mnist.pkl"), "rb") as f:
    MODEL = pickle.load(f)
with open(os.path.join(BASE, "model", "scaler.pkl"), "rb") as f:
    SCALER = pickle.load(f)

print("✅ Model and scaler loaded.")


def preprocess_canvas(data_url: str) -> np.ndarray:
    """
    Convert a base64 PNG canvas image → 28×28 normalized feature vector
    that matches the training data format.
    """
    # Decode base64 PNG
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale

    # Invert: canvas is white digit on black; training data is white digit
    # sklearn digits: 0=black background, 16=white digit
    # Our canvas: 255=white bg, 0=black stroke → invert
    img = ImageOps.invert(img)

    # Find bounding box of the digit and crop with padding
    bbox = img.getbbox()
    if bbox:
        # Add 20% padding around the digit
        w, h = img.size
        pad = int(max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * 0.25)
        bbox = (
            max(0, bbox[0]-pad),
            max(0, bbox[1]-pad),
            min(w, bbox[2]+pad),
            min(h, bbox[3]+pad)
        )
        img = img.crop(bbox)

    # Apply slight blur to smooth stroke edges
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    # Resize to 8×8 (match training data resolution) then upscale to 28×28
    img_8 = img.resize((8, 8), Image.LANCZOS)
    img_arr_8 = np.array(img_8, dtype=np.float64)

    # Upscale 8→28 the same way training did
    img_arr_28 = zoom(img_arr_8, 28/8, order=1).flatten()

    # Normalize
    img_arr_28 = img_arr_28.reshape(1, -1)
    img_arr_28 = SCALER.transform(img_arr_28)
    return img_arr_28


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data"}), 400

    try:
        features = preprocess_canvas(data["image"])
        prediction = int(MODEL.predict(features)[0])
        probas = MODEL.predict_proba(features)[0]

        # Top-3 predictions with confidence
        top3_idx = np.argsort(probas)[::-1][:3]
        top3 = [
            {"digit": int(i), "confidence": round(float(probas[i]) * 100, 1)}
            for i in top3_idx
        ]

        return jsonify({
            "prediction": prediction,
            "confidence": round(float(probas[prediction]) * 100, 1),
            "top3": top3
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stats")
def stats():
    return jsonify({
        "model": "K-Nearest Neighbors (KNN)",
        "best_k": MODEL.n_neighbors,
        "weights": MODEL.weights,
        "algorithm": MODEL.algorithm,
        "accuracy": "98.33%",
        "cv_accuracy": "98.54% ± 0.92%",
        "dataset": "sklearn digits (1797 samples, upscaled 8×8 → 28×28)",
        "features": 784,
        "classes": 10
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
