"""
server.py
=========
Flask Backend API Server for Astro Image Quality Inspection (V2 Engine).
Serves the Frontend Dashboard and provides /api/analyze endpoint for real-time inference.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# Ensure local directory is in Python path to import pipeline modules
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from astro_pipeline_v2 import (
    CLASS_NAMES,
    DEFAULT_THRESHOLDS_V2,
    extract_features_v2,
    score_rules_v2,
    AstroFeaturesV2,
)

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

ONNX_MODEL_PATH = BASE_DIR / "eff_b0_kfold_add_focal_r2.onnx"

# Pre-load ONNX session globally for sub-30ms in-memory inference
onnx_session = None
input_meta = None
if ONNX_MODEL_PATH.exists():
    try:
        import onnxruntime as ort
        onnx_session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
        input_meta = onnx_session.get_inputs()[0]
        print(f"[OK] Preloaded ONNX Model: {ONNX_MODEL_PATH.name}")
    except Exception as e:
        print(f"[WARN] Failed to preload ONNX model: {e}")

app = Flask(__name__, static_folder=str(BASE_DIR / "static"), static_url_path="")


@app.after_request
def add_cache_control(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "engine": "Astro Physics-Based Pipeline V2",
        "onnx_model_loaded": onnx_session is not None,
        "classes": CLASS_NAMES,
    })


ASTRO_MEAN = np.array([0.353340744972229, 0.353340744972229, 0.353340744972229], dtype=np.float32)
ASTRO_STD = np.array([0.13850915431976318, 0.13850915431976318, 0.13850915431976318], dtype=np.float32)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please send image in 'file' field."}), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        file_bytes = uploaded_file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if bgr is None:
            return jsonify({"error": "Failed to decode image. Ensure file is a valid PNG, JPG, or WebP."}), 400

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Physics Feature Extraction (V2)
        features: AstroFeaturesV2 = extract_features_v2(gray)

        # 2. Physics Rule Scoring (V2)
        rule_scores = score_rules_v2(features)
        top_rule_class = max(rule_scores, key=rule_scores.get)
        top_rule_score = rule_scores[top_rule_class]

        rule_details = {}
        for cname, sc in rule_scores.items():
            th = DEFAULT_THRESHOLDS_V2.get(cname, 0.50)
            rule_details[cname] = {
                "score": round(float(sc), 4),
                "score_pct": round(float(sc) * 100.0, 2),
                "threshold": th,
                "threshold_pct": round(th * 100.0, 1),
                "passed": bool(sc >= th),
            }

        # 3. Fast In-Memory CNN Model Inference (Exact Training Normalization)
        cnn_result = {
            "available": False,
            "predicted_class": "N/A",
            "confidence_pct": 0.0,
            "probabilities_pct": {},
        }
        if onnx_session is not None and input_meta is not None:
            try:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
                norm = (resized.astype(np.float32) / 255.0 - ASTRO_MEAN) / ASTRO_STD
                input_tensor = np.transpose(norm, (2, 0, 1))[None, ...]

                outputs = onnx_session.run(None, {input_meta.name: input_tensor})
                raw = np.asarray(outputs[0]).reshape(-1)
                probs = raw if np.isclose(np.sum(raw), 1.0, atol=1e-3) and np.all(raw >= 0) else softmax(raw)
                preds = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}

                cnn_top = max(preds, key=preds.get)
                cnn_result = {
                    "available": True,
                    "predicted_class": cnn_top,
                    "confidence": round(float(preds[cnn_top]), 4),
                    "confidence_pct": round(float(preds[cnn_top]) * 100.0, 2),
                    "probabilities_pct": {k: round(float(v) * 100.0, 2) for k, v in preds.items()},
                }
            except Exception as e:
                cnn_result["error"] = str(e)

        # 4. Model Consensus
        consensus = "UNKNOWN"
        if cnn_result["available"]:
            consensus = "MATCH" if top_rule_class == cnn_result["predicted_class"] else "DISAGREE"

        # 5. Build Response Payload
        response = {
            "filename": uploaded_file.filename,
            "dimensions": {"width": w, "height": h},
            "status": "SUCCESS",
            "consensus": consensus,
            "rule_based": {
                "top_class": top_rule_class,
                "top_score_pct": round(top_rule_score * 100.0, 2),
                "scores": rule_details,
            },
            "cnn_model": cnn_result,
            "physics_metrics": {
                "star_count": features.star_count,
                "median_fwhm": round(features.median_fwhm, 2),
                "mean_fwhm": round(features.mean_fwhm, 2),
                "median_eccentricity": round(features.median_eccentricity, 3),
                "mean_aspect_ratio": round(features.mean_aspect_ratio, 2),
                "mean_circularity": round(features.mean_circularity, 3),
                "mean_hollowness": round(features.mean_hollowness, 3),
                "elongated_star_count": features.elongated_star_count,
                "elongated_angle_consistency": round(features.elongated_angle_consistency, 3),
                "max_streak_length_ratio": round(features.max_streak_length_ratio, 3),
                "max_streak_aspect_ratio": round(features.max_streak_aspect_ratio, 2),
                "max_projection_diff": round(features.max_projection_diff, 2),
                "saturated_ratio": round(features.saturated_ratio, 4),
                "background_median": round(features.background_median, 1),
                "background_mad": round(features.background_mad, 2),
                "sharpness": round(features.sharpness, 1),
            },
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Server processing error: {str(e)}"}), 500


def main():
    parser = argparse.ArgumentParser(description="Astro Quality Web API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind.")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f" ASTRO IMAGE QUALITY INSPECTION SERVER (V2 ENGINE)")
    print(f" Running at: http://{args.host}:{args.port}")
    print(f" Web UI    : http://{args.host}:{args.port}/")
    print(f" API Health: http://{args.host}:{args.port}/api/health")
    print(f"=======================================================\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
