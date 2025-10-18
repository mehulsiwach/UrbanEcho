# app/app.py
import os
import io
import json
import numpy as np
from datetime import datetime, timezone
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import pickle
import firebase_admin
from firebase_admin import credentials, firestore
from yamnet_utils import wavfile_to_mean_embedding

# ---------- Paths & env ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
os.makedirs(STATIC_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "yamnet_noise_classifier.h5")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Firestore init:
# Set env var GOOGLE_APPLICATION_CREDENTIALS to path of service account JSON OR
# provide path here:

# Determine Firebase credentials path:
# 1) prefer env var GOOGLE_APPLICATION_CREDENTIALS
# 2) then /app/serviceAccount.json (copied into image by Dockerfile)
# 3) then fall back to your local Windows path (dev only)
FIREBASE_CRED_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "/app/serviceAccount.json" or r"C:\Users\mehul\Downloads\Urban\serviceAccount.json"
if not FIREBASE_CRED_PATH or not os.path.exists(FIREBASE_CRED_PATH):
    raise RuntimeError("Firebase credentials file not found at: " + str(FIREBASE_CRED_PATH))
cred = credentials.Certificate(FIREBASE_CRED_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()
COLLECTION_NAME = "noise_data"  # Firestore collection

# ---------- Load models ----------
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    raise RuntimeError(f"Place your model and encoder in {MODEL_DIR} as yamnet_noise_classifier.h5 and label_encoder.pkl")

model = load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# Simple mapping class -> approximate dB for visualization
NOISE_DB_MAP = {
    "air_conditioner": 50, "car_horn": 90, "children_playing": 60,
    "dog_bark": 70, "drilling": 95, "engine_idling": 80,
    "gun_shot": 120, "jackhammer": 100, "siren": 110, "street_music": 75
}

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

# ---------- Helper: Predict ----------
def predict_from_fileobj(fileobj):
    # get embedding
    emb = wavfile_to_mean_embedding(fileobj)
    if emb is None:
        return None, None, None
    preds = model.predict(np.expand_dims(emb, axis=0))[0]
    idx = int(np.argmax(preds))
    label = label_encoder.inverse_transform([idx])[0]
    conf = float(preds[idx])

    # Calculate noise level from audio
    import librosa
    fileobj.seek(0)  # rewind file pointer
    y, sr = librosa.load(fileobj, sr=16000, mono=True)
    rms = np.sqrt(np.mean(y**2))
    noise_db = round(20 * np.log10(rms + 1e-6) + 94)  # 94 dB SPL reference

    return label, conf, noise_db

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """
    Expects form-data:
      - audio_file (file)
      - lat (string float)
      - lon (string float)
    """
    audio = request.files.get("audio_file")
    lat = request.form.get("lat")
    lon = request.form.get("lon")

    # Validate lat/lon; if missing fallback to None (we don't store bad coordinates)
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        lat = None
        lon = None

    if audio is None:
        return jsonify({"error": "no audio"}), 400

    # Predict
    label, conf, noise_db = predict_from_fileobj(audio)
    if label is None:
        return jsonify({"error": "prediction failed"}), 500

    # Firestore document
    doc = {
        "label": label,
        "confidence": conf,
        "noise_db": float(noise_db),
        "timestamp": datetime.now(timezone.utc)
    }               
    if lat is not None and lon is not None:
        doc["lat"] = float(lat)
        doc["lon"] = float(lon)
    else:
        # If no coords, do not add lat/lon (optional: you can assign default or skip storing)
        doc["lat"] = None
        doc["lon"] = None

    # Write to Firestore
    db.collection(COLLECTION_NAME).add(doc)

    # Return payload for UI
    return jsonify({
        "label": label,
        "confidence": round(conf * 100, 2),
        "noise_db": noise_db
    }), 200

@app.route("/data")
def data():
    points = []
    try:
        docs = db.collection(COLLECTION_NAME).stream()
    except Exception as e:
        return jsonify([])

    for d in docs:
        r = d.to_dict() or {}
        lat = r.get("lat")
        lon = r.get("lon")
        if lat is None or lon is None:
            continue

        # pick label from known keys, normalize to string
        label = r.get("label") or r.get("pred_label") or r.get("noise_type") or "unknown"
        label = str(label)

        noise_db = r.get("noise_db")
        try:
            noise_db = float(noise_db) if noise_db is not None else None
        except Exception:
            noise_db = None

        ts = r.get("timestamp")
        # Firestore Timestamp -> ISO string, fallback to str()
        try:
            timestamp = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        except Exception:
            timestamp = str(ts)

        points.append({
            "lat": float(lat),
            "lon": float(lon),
            "noise_db": noise_db,
            "label": label,
            "timestamp": timestamp
        })

    return jsonify(points), 200

# ---------- Run ----------
if __name__ == "__main__":
    # debug only; for production use gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)
