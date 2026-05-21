import cv2
import time

FIRE_STATE = {
    "is_fire": False,
    "predictions": None,
    "best_model": None
}
def find_available_camera():
    """Hàm tự động quét và tìm Camera đang hoạt động"""
    for cam_id in range(6):
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"✅ Đã chốt Camera ID = {cam_id} cho Web Stream")
                return cap
            cap.release()
    return None


def generate_frames(trainer, yolo_model_instance):
    """
    Hàm Generator: Liên tục đọc khung hình, YOLO phân tích và đẩy lên Web.
    Nhận 'yolo_model_instance' từ Web App truyền sang để phân tích.
    """
    cap = find_available_camera()
    if cap is None:
        print("❌ Không tìm thấy Camera cho Web Stream!")
        return

    temp_img_path = "temp_webcam_eval.jpg"
    last_ml_time = 0  # Bộ đếm thời gian để không làm lag máy

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        current_time = time.time()
        is_fire_yolo = False

        # Nếu YOLO đã sẵn sàng thì đưa vào phân tích
        if yolo_model_instance and yolo_model_instance.is_trained:
            # Phân tích frame hiện tại
            results = yolo_model_instance.model.predict(source=frame, conf=0.5, verbose=False)[0]

            probs = results.probs
            top1_idx = probs.top1
            confidence = probs.top1conf.item()
            class_name = results.names[top1_idx].upper()

            is_fire = 'FIRE' in class_name and 'NO' not in class_name
            FIRE_STATE["is_fire"] = is_fire

            # Đổi màu xanh/đỏ
            color = (0, 0, 255) if is_fire else (0, 255, 0)

            # Viết text lên frame
            cv2.putText(frame, f"STATUS: {class_name} ({confidence * 100:.1f}%)",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if current_time - last_ml_time >= 1.0:
                all_preds = {}

                # 1. Ghi nhận điểm của YOLO
                class_names_dict = results.names
                fire_idx = 0 if 'FIRE' in class_names_dict[0].upper() else 1
                no_fire_idx = 1 - fire_idx

                all_preds['YOLO'] = {
                    'prediction': 'FIRE' if is_fire_yolo else 'NO FIRE',
                    'confidence': confidence,
                    'probability_fire': float(probs.data.cpu().numpy()[fire_idx]),
                    'probability_no_fire': float(probs.data.cpu().numpy()[no_fire_idx])
                }

                # 2. Ghi nhận điểm của 5 Models ML truyền thống
                if trainer and trainer.trained_models:
                    cv2.imwrite(temp_img_path, frame)
                    try:
                        ml_preds = trainer.predict_single_image(temp_img_path)
                        all_preds.update(ml_preds)
                    except Exception as e:
                        pass  # Bỏ qua nếu có lỗi nhỏ lúc trích xuất ảnh

                # 3. Cập nhật ra toàn cục cho Web lấy dữ liệu
                if all_preds:
                    FIRE_STATE["predictions"] = all_preds
                    best = max(all_preds.items(), key=lambda x: x[1]['confidence'])
                    FIRE_STATE["best_model"] = {"name": best[0]}

                last_ml_time = current_time  # Reset bộ đếm

                # --- Vẽ UI cơ bản lên Camera ---
            color = (0, 0, 255) if is_fire_yolo else (0, 255, 0)
            cv2.putText(frame, f"YOLO STATUS: {class_name}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


            if is_fire:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                # Chớp nháy chữ cảnh báo
                if int(time.time() * 2) % 2 == 0:
                    cv2.putText(frame, "!!! FIRE ALARM !!!",
                                (frame.shape[1] // 2 - 120, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Chuyển đổi frame thành chuẩn ảnh JPEG dạng bytes để stream qua HTTP
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Dùng kỹ thuật yield (trả về liên tục) của Python
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')