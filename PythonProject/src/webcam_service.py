#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Webcam Service — Stream video real-time và phát hiện lửa.

Cải tiến so với phiên bản cũ:
 1. Temporal Consistency: Cần >=N frame liên tiếp mới xác nhận cháy
    → Loại false positive do da người / hoàng hôn thoáng qua
 2. Fire Area Filter: Vùng màu lửa phải chiếm >= X% diện tích frame
    → Loại false positive do vùng màu cam/đỏ nhỏ (tay vẫy, ánh đèn)
 3. Skin Exclusion Mask: Loại bỏ vùng màu da người khỏi fire mask
    (S ngưỡng cao hơn và hue không quá vàng)
 4. Camera Release: finally block đảm bảo camera luôn được giải phóng
 5. ES Logging: Ghi lịch sử khi fire được xác nhận lần đầu
"""

import cv2
import time
import os
import numpy as np
from alert_service import trigger_telegram_alert

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Cấu hình chống false positive ───────────────────────────
FIRE_CONFIRMATION_FRAMES = int(os.getenv("FIRE_CONFIRMATION_FRAMES", "5"))
MIN_FIRE_AREA_RATIO      = float(os.getenv("MIN_FIRE_AREA_RATIO", "0.02"))

# Vùng màu da người trong HSV (để loại trừ)
# Da người: H 0-25°, S thấp (20-170), V đa dạng
_SKIN_HSV_LO = np.array([0,  20,  70])
_SKIN_HSV_HI = np.array([25, 170, 255])

# ── State toàn cục (frontend đọc qua /get_fire_status) ──────
FIRE_STATE = {
    "is_fire":         False,  # YOLO raw detection
    "is_confirmed":    False,  # Đã qua temporal check
    "fire_frame_count": 0,     # Số frame liên tiếp detect fire
    "threshold":       FIRE_CONFIRMATION_FRAMES,
    "predictions":     None,
    "best_model":      None,
}

# ── Internal counters ────────────────────────────────────────
_fire_frame_counter: int = 0
_was_confirmed:      bool = False   # Track state change để log ES


# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def find_available_camera():
    """Tự động quét và mở camera đầu tiên hoạt động được."""
    cam_id_env = os.getenv("CAMERA_ID", "auto")

    if cam_id_env != "auto":
        try:
            cap = cv2.VideoCapture(int(cam_id_env), cv2.CAP_DSHOW)
            if cap.isOpened() and cap.read()[0]:
                print(f"✅ Camera ID={cam_id_env} (từ .env)")
                return cap
            cap.release()
        except Exception:
            pass

    for cam_id in range(6):
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"✅ Camera ID={cam_id} (auto-detect)")
                return cap
            cap.release()
    return None


def _check_fire_area(frame: np.ndarray) -> tuple:
    """
    Kiểm tra diện tích vùng màu lửa trong frame.
    Sử dụng ngưỡng S >= 175 (cao hơn extractor gốc 150) để loại da người.
    Áp thêm skin exclusion mask.

    Returns:
        (has_enough_area: bool, fire_ratio: float)
    """
    if frame is None:
        return False, 0.0

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    total = frame.shape[0] * frame.shape[1]

    # Fire color masks — S >= 175 để lọc da người và ánh đèn nhạt
    m_red1   = cv2.inRange(hsv, np.array([0,   175, 150]), np.array([10,  255, 255]))
    m_red2   = cv2.inRange(hsv, np.array([170, 175, 150]), np.array([180, 255, 255]))
    m_orange = cv2.inRange(hsv, np.array([10,  175, 150]), np.array([25,  255, 255]))
    m_yellow = cv2.inRange(hsv, np.array([25,  150, 200]), np.array([35,  255, 255]))

    fire = cv2.bitwise_or(m_red1, m_red2)
    fire = cv2.bitwise_or(fire, m_orange)
    fire = cv2.bitwise_or(fire, m_yellow)

    # Loại trừ vùng màu da người
    skin = cv2.inRange(hsv, _SKIN_HSV_LO, _SKIN_HSV_HI)
    fire = cv2.bitwise_and(fire, cv2.bitwise_not(skin))

    ratio = float(np.sum(fire > 0)) / total
    return ratio >= MIN_FIRE_AREA_RATIO, ratio


# ════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ════════════════════════════════════════════════════════════

def generate_frames(trainer, yolo_model_instance):
    """
    Generator: đọc frame liên tục, phân tích, stream qua HTTP.
    Camera luôn được giải phóng trong finally block.
    """
    global _fire_frame_counter, _was_confirmed

    cap = find_available_camera()
    if cap is None:
        print("❌ Không tìm thấy Camera!")
        return

    temp_img = "temp_webcam_eval.jpg"
    last_ml_time = 0.0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            now   = time.time()

            # ── YOLO inference ───────────────────────────────
            if yolo_model_instance and yolo_model_instance.is_trained:
                results    = yolo_model_instance.model.predict(
                    source=frame, conf=0.5, verbose=False
                )[0]
                probs      = results.probs
                top1_idx   = probs.top1
                confidence = probs.top1conf.item()
                class_name = results.names[top1_idx].upper()

                yolo_fire  = "FIRE" in class_name and "NO" not in class_name

                # ── Area + skin filter ───────────────────────
                has_area, fire_ratio = _check_fire_area(frame)
                raw_fire = yolo_fire and has_area

                # ── Temporal consistency ─────────────────────
                if raw_fire:
                    _fire_frame_counter += 1
                else:
                    # Giảm dần thay vì reset cứng → ổn định hơn
                    _fire_frame_counter = max(0, _fire_frame_counter - 1)

                is_confirmed = _fire_frame_counter >= FIRE_CONFIRMATION_FRAMES

                # ── Cập nhật global state ────────────────────
                FIRE_STATE.update({
                    "is_fire":          raw_fire,
                    "is_confirmed":     is_confirmed,
                    "fire_frame_count": _fire_frame_counter,
                    "threshold":        FIRE_CONFIRMATION_FRAMES,
                })

                # ── Alert & ES log khi lần đầu confirmed ────
                if is_confirmed and not _was_confirmed:
                    trigger_telegram_alert(frame, confidence)
                    _log_webcam_event(FIRE_STATE, confidence)

                _was_confirmed = is_confirmed

                # ── Overlay text ─────────────────────────────
                if is_confirmed:
                    color = (0, 0, 255)
                    label = f"🔥 CONFIRMED FIRE  {confidence*100:.0f}%"
                elif raw_fire:
                    color = (0, 140, 255)
                    label = f"Detecting... [{_fire_frame_counter}/{FIRE_CONFIRMATION_FRAMES}]  {fire_ratio*100:.1f}%"
                else:
                    color = (0, 220, 0)
                    label = f"{class_name}  {confidence*100:.0f}%"

                cv2.putText(frame, label, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
                cv2.putText(frame, f"fire area: {fire_ratio*100:.1f}%",
                            (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

                # Boder cảnh báo
                if is_confirmed:
                    cv2.rectangle(frame, (0, 0),
                                  (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                    if int(now * 2) % 2 == 0:
                        cv2.putText(frame, "!!! FIRE ALARM !!!",
                                    (frame.shape[1]//2 - 160, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                elif raw_fire:
                    cv2.rectangle(frame, (0, 0),
                                  (frame.shape[1], frame.shape[0]), (0, 140, 255), 5)

                # ── ML models prediction mỗi giây ───────────
                if now - last_ml_time >= 1.0:
                    _update_ml_predictions(frame, results, probs, yolo_fire, trainer, temp_img)
                    last_ml_time = now

            # ── Encode → JPEG bytes ──────────────────────────
            ret, buf = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    finally:
        cap.release()
        print("📷 Camera đã được giải phóng.")


def _update_ml_predictions(frame, yolo_results, probs, yolo_fire, trainer, temp_img):
    """Cập nhật FIRE_STATE với prediction từ YOLO + 5 ML models."""
    class_names_dict = yolo_results.names
    fire_idx    = 0 if "FIRE" in class_names_dict[0].upper() else 1
    no_fire_idx = 1 - fire_idx

    all_preds = {
        "YOLO": {
            "prediction":       "FIRE" if yolo_fire else "NO FIRE",
            "confidence":       float(probs.top1conf.item()),
            "probability_fire": float(probs.data.cpu().numpy()[fire_idx]),
            "probability_no_fire": float(probs.data.cpu().numpy()[no_fire_idx]),
        }
    }

    if trainer and trainer.trained_models:
        cv2.imwrite(temp_img, frame)
        try:
            ml_preds = trainer.predict_single_image(temp_img)
            all_preds.update(ml_preds)
        except Exception:
            pass

    if all_preds:
        best = max(all_preds.items(), key=lambda x: x[1]["confidence"])
        FIRE_STATE["predictions"] = all_preds
        FIRE_STATE["best_model"]  = {
            "name":       best[0],
            "prediction": best[1]["prediction"],
            "confidence": best[1]["confidence"],
        }


def _log_webcam_event(state: dict, confidence: float):
    """Ghi sự kiện confirmed fire vào Elasticsearch / JSON file."""
    try:
        from elasticsearch_service import get_logger
        preds    = state.get("predictions") or {}
        best_mdl = state.get("best_model") or {
            "name": "YOLO", "prediction": "FIRE", "confidence": confidence
        }
        get_logger().log_event(
            source="webcam",
            predictions=preds,
            best_model={**best_mdl, "prediction": "FIRE"},
            is_confirmed=True,
        )
    except Exception as exc:
        print(f"⚠️  [Webcam] ES log lỗi: {exc}")