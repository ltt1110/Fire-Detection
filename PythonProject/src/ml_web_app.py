#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Application — ML + YOLO Fire Detection
==========================================
Cải tiến:
  - Elasticsearch / JSON history logging
  - /history, /statistics, /config endpoints
  - load_latest_models() sort đúng theo mtime
  - Cleanup uploads folder cũ (>24h)
  - Dynamic fire_threshold qua /config API
"""

import os
import uuid
import json
import base64
import glob
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, Response

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ml_models import MLModelTrainer
from fire_feature_extractor import FireFeatureExtractor
from yolo_model import YOLOFireDetector, YOLO_AVAILABLE
from webcam_service import generate_frames, FIRE_STATE
from elasticsearch_service import get_logger as get_es_logger

# ── Flask app ────────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"]       = "uploads"
app.config["MAX_CONTENT_LENGTH"]  = 16 * 1024 * 1024
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ── Global instances ─────────────────────────────────────────
trainer           = MLModelTrainer()
feature_extractor = FireFeatureExtractor()
es_logger         = get_es_logger()

# ── Dynamic config (có thể thay đổi qua /config API) ────────
app_config = {
    "fire_threshold": float(os.getenv("FIRE_THRESHOLD", "0.55")),
}

# ── YOLO ────────────────────────────────────────────────────
yolo_detector = None
if YOLO_AVAILABLE:
    try:
        yolo_detector = YOLOFireDetector()
        print("✅ YOLO detector sẵn sàng")
    except Exception as exc:
        print(f"⚠️  YOLO init lỗi: {exc}")
else:
    print("⚠️  YOLO chưa cài. Chạy: pip install ultralytics")


# ════════════════════════════════════════════════════════════
# MODEL LOADERS
# ════════════════════════════════════════════════════════════

def load_latest_models() -> bool:
    """Load bộ models mới nhất (sort theo mtime → chính xác)."""
    models_dir = "trained_models"
    if not os.path.exists(models_dir):
        print("❌ Thư mục trained_models không tồn tại")
        return False

    model_files = [
        f for f in os.listdir(models_dir)
        if f.endswith(".pkl") and not f.startswith("scaler")
    ]
    if not model_files:
        print("❌ Không tìm thấy model .pkl")
        return False

    # Sort theo modification time — mới nhất trước
    model_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(models_dir, f)),
        reverse=True,
    )
    first = model_files[0]
    parts = first.split("_")
    if len(parts) >= 3:
        timestamp = f"{parts[-2]}_{parts[-1].replace('.pkl', '')}"
    elif len(parts) >= 2:
        timestamp = parts[-1].replace(".pkl", "")
    else:
        print("❌ Không parse được timestamp")
        return False

    print(f"🔍 Timestamp mới nhất: {timestamp}")
    try:
        trainer.load_models(timestamp)
        print(f"✅ Đã load {len(trainer.trained_models)} models")
        return True
    except Exception as exc:
        print(f"❌ Load models lỗi: {exc}")
        return False


def load_latest_yolo_model() -> bool:
    """Load YOLO model mới nhất từ trained_models/yolo/."""
    global yolo_detector
    if not YOLO_AVAILABLE:
        return False

    yolo_dir = "trained_models/yolo"
    if not os.path.exists(yolo_dir):
        return False

    best_files = glob.glob(os.path.join(yolo_dir, "**", "best.pt"), recursive=True)
    if not best_files:
        return False

    best_files.sort(key=os.path.getmtime, reverse=True)
    try:
        yolo_detector = YOLOFireDetector(model_path=best_files[0])
        print(f"✅ YOLO model: {best_files[0]}")
        return True
    except Exception as exc:
        print(f"❌ YOLO load lỗi: {exc}")
        return False


def cleanup_old_uploads(max_age_hours: int = 24):
    """Xóa ảnh upload cũ hơn max_age_hours."""
    folder   = app.config["UPLOAD_FOLDER"]
    cutoff   = time.time() - max_age_hours * 3600
    deleted  = 0
    for f in os.listdir(folder):
        fpath = os.path.join(folder, f)
        if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
            try:
                os.remove(fpath)
                deleted += 1
            except Exception:
                pass
    if deleted:
        print(f"🗑️  Đã xóa {deleted} ảnh upload cũ")


# ── Khởi động ────────────────────────────────────────────────
print("🚀 Khởi động Fire Detection Web App...")
print(f"📊 ES status: {es_logger.status}")
load_latest_models()
load_latest_yolo_model()
cleanup_old_uploads()


# ════════════════════════════════════════════════════════════
# ROUTES — Existing
# ════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("ml_index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Không có file"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Chưa chọn file"})

    if not trainer.trained_models:
        return jsonify({"error": "Chưa có models. Vui lòng load models trước."})

    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        threshold   = app_config["fire_threshold"]
        predictions = trainer.predict_single_image(filepath, threshold=threshold)

        if yolo_detector and yolo_detector.is_trained:
            try:
                yolo_pred = yolo_detector.predict(filepath)
                predictions["YOLO"] = yolo_pred
            except Exception as exc:
                print(f"⚠️  YOLO predict lỗi: {exc}")

        # Chuẩn hóa numpy types
        def _convert(obj):
            if isinstance(obj, np.generic):   return obj.item()
            if isinstance(obj, np.ndarray):   return obj.tolist()
            if isinstance(obj, dict):         return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):         return [_convert(v) for v in obj]
            return obj

        predictions = _convert(predictions)
        best_model  = max(predictions.items(), key=lambda x: x[1]["confidence"])
        best_info   = {
            "name":       best_model[0],
            "prediction": best_model[1]["prediction"],
            "confidence": best_model[1]["confidence"],
        }

        # Log vào ES / JSON
        es_logger.log_event(
            source="upload",
            predictions=predictions,
            best_model=best_info,
            image_filename=filename,
        )

        with open(filepath, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")

        return jsonify({
            "success":    True,
            "image":      encoded,
            "predictions": predictions,
            "best_model": best_info,
            "timestamp":  datetime.now().isoformat(),
            "threshold_used": threshold,
        })

    except Exception as exc:
        return jsonify({"error": f"Lỗi phân tích: {str(exc)}"})


@app.route("/uploaded_file/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/health")
def health_check():
    ml_loaded   = len(trainer.trained_models) > 0
    yolo_status = bool(yolo_detector and yolo_detector.is_trained) if YOLO_AVAILABLE else False
    model_names = list(trainer.trained_models.keys())
    if yolo_status:
        model_names.append("YOLO")
    return jsonify({
        "status":          "healthy",
        "models_loaded":   ml_loaded,
        "num_models":      len(trainer.trained_models),
        "model_names":     model_names,
        "yolo_available":  YOLO_AVAILABLE,
        "yolo_loaded":     yolo_status,
        "fire_threshold":  app_config["fire_threshold"],
        "es_connected":    es_logger.use_es,
    })


@app.route("/models")
def get_models():
    info = {}
    for name, model in trainer.trained_models.items():
        info[name] = {
            "type":       type(model).__name__,
            "parameters": str(model.get_params()) if hasattr(model, "get_params") else "N/A",
        }
    if yolo_detector and yolo_detector.is_trained:
        yi = yolo_detector.get_model_info()
        info["YOLO"] = {
            "type":        yi["model_type"],
            "parameters":  f"{yi['parameters']:,}",
            "is_trained":  yi["is_trained"],
            "model_path":  yi["model_path"],
        }
    return jsonify({
        "models_loaded": len(info),
        "models":        info,
    })


@app.route("/load-models", methods=["POST"])
def load_models_endpoint():
    data      = request.get_json() or {}
    timestamp = data.get("timestamp")
    if not timestamp:
        return jsonify({"error": "Thiếu timestamp"})
    try:
        if timestamp == "latest":
            success = load_latest_models()
        else:
            success = trainer.load_models(timestamp)
        if success:
            return jsonify({
                "success": True,
                "message": "Đã load models",
                "models":  list(trainer.trained_models.keys()),
            })
        return jsonify({"error": "Không thể load models"})
    except Exception as exc:
        return jsonify({"error": str(exc)})


@app.route("/train-status")
def train_status():
    models_dir = "trained_models"
    if not os.path.exists(models_dir):
        return jsonify({"status": "no_models", "message": "Chưa train"})

    pkl_files = [
        f for f in os.listdir(models_dir)
        if f.endswith(".pkl") and not f.startswith("scaler")
    ]
    if not pkl_files:
        return jsonify({"status": "no_models", "message": "Không tìm thấy"})

    timestamps = sorted(
        set(
            f"{p[-2]}_{p[-1].replace('.pkl','')}"
            for f in pkl_files
            if len((p := f.split("_"))) >= 3
        ),
        reverse=True,
    )
    return jsonify({
        "status":                "models_available",
        "available_timestamps":  timestamps,
        "latest_timestamp":      timestamps[0] if timestamps else None,
        "models_loaded":         len(trainer.trained_models) > 0,
        "num_models_loaded":     len(trainer.trained_models),
        "model_names":           list(trainer.trained_models.keys()),
    })


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(trainer, yolo_detector),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/get_fire_status")
def get_fire_status():
    return jsonify(FIRE_STATE)


# ════════════════════════════════════════════════════════════
# ROUTES — New
# ════════════════════════════════════════════════════════════

@app.route("/history")
def history():
    """Trả về lịch sử phát hiện từ ES / JSON file."""
    limit   = int(request.args.get("limit",   50))
    source  = request.args.get("source",  None)  # "upload"|"webcam"|None
    is_fire = request.args.get("is_fire", None)
    hours   = int(request.args.get("hours", 24))

    if is_fire is not None:
        is_fire = is_fire.lower() == "true"

    events = es_logger.get_recent_events(
        limit=limit, source=source, is_fire=is_fire, hours=hours
    )
    return jsonify({
        "events":      events,
        "count":       len(events),
        "data_source": "elasticsearch" if es_logger.use_es else "json_file",
    })


@app.route("/statistics")
def statistics():
    """Thống kê tổng hợp từ ES / JSON file."""
    hours = int(request.args.get("hours", 24))
    stats = es_logger.get_statistics(hours=hours)
    return jsonify(stats)


@app.route("/config", methods=["GET", "POST"])
def config():
    """GET: lấy cấu hình hiện tại.  POST: cập nhật fire_threshold."""
    if request.method == "GET":
        return jsonify({
            **app_config,
            "es_status": es_logger.status,
        })

    data = request.get_json() or {}

    if "fire_threshold" in data:
        val = float(data["fire_threshold"])
        if 0.10 <= val <= 0.99:
            app_config["fire_threshold"] = round(val, 2)
        else:
            return jsonify({"error": "fire_threshold phải trong [0.10, 0.99]"}), 400

    return jsonify({"success": True, "config": app_config})


@app.route("/es-status")
def es_status():
    """Trạng thái kết nối Elasticsearch."""
    return jsonify(es_logger.status)


# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port  = int(os.getenv("FLASK_PORT", "8085"))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    print(f"🌐 Truy cập: http://localhost:{port}")
    app.run(debug=debug, host="0.0.0.0", port=port)