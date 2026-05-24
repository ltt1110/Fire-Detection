#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alert Service — Gửi cảnh báo qua Telegram khi phát hiện cháy.
Token và Chat ID được đọc từ file .env (bảo mật).
"""

import cv2
import time
import requests
from threading import Thread

# Load biến môi trường
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os

# ── Cấu hình Telegram (đọc từ .env) ─────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN",   "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
COOLDOWN_TIME    = int(os.getenv("TELEGRAM_COOLDOWN", "20"))  # giây

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("⚠️  [Alert] Telegram chưa cấu hình. Kiểm tra file .env")

# Thời gian gửi cuối (chống spam)
_last_alert_time: float = 0


def _send_telegram_worker(frame, confidence: float):
    """Worker thread — nén ảnh và gửi qua Telegram API."""
    try:
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            return
        image_bytes = buffer.tobytes()

        url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        caption = (
            f"🚨 CẢNH BÁO CHÁY KHẨN CẤP!\n"
            f"🔥 Độ tin cậy (Ensemble): {confidence * 100:.1f}%\n"
            f"⏰ Thời gian: {time.strftime('%H:%M:%S %d/%m/%Y')}\n"
            f"📍 Vị trí: {os.getenv('CAMERA_LOCATION', 'Camera AI')}"
        )

        response = requests.post(
            url,
            files={"photo": ("alert.jpg", image_bytes, "image/jpeg")},
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
            timeout=15,
        )

        if response.status_code == 200:
            print("📲 Gửi cảnh báo Telegram thành công!")
        else:
            print(f"❌ Telegram lỗi {response.status_code}: {response.text[:200]}")

    except Exception as exc:
        print(f"❌ Lỗi gửi Telegram: {exc}")


def trigger_telegram_alert(frame, confidence: float):
    """
    Kiểm tra cooldown rồi mở thread ngầm để gửi cảnh báo.
    Thread daemon → tự kết thúc khi app thoát.
    """
    global _last_alert_time
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    now = time.time()
    if now - _last_alert_time < COOLDOWN_TIME:
        return

    _last_alert_time = now
    thread = Thread(
        target=_send_telegram_worker,
        args=(frame.copy(), confidence),
        daemon=True,
    )
    thread.start()