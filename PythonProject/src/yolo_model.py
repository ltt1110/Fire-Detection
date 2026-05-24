#!/usr/bin/env python3
"""
YOLO Model for Fire Detection
Sử dụng YOLOv8 Classification để so sánh với các mô hình ML truyền thống
"""

import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Any
import shutil

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except (ImportError, OSError, Exception) as _yolo_err:
    YOLO_AVAILABLE = False
    print(f"[YOLO] Khong the load: {_yolo_err}")
    print("[YOLO] He thong se chay khong co YOLO, chi dung ML models.")


class YOLOFireDetector:
    """YOLO model cho Fire Detection"""
    
    def __init__(self, model_path: str = None):
        """
        Khởi tạo YOLO model
        
        Args:
            model_path: Đường dẫn đến model đã train, nếu None sẽ dùng pretrained
        """
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO chưa được cài đặt")
        
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            print(f"📂 Loading YOLO model từ: {model_path}")
            self.model = YOLO(model_path)
            self.is_trained = True
            print("✅ YOLO model đã được load!")
        else:
            print("📦 Sử dụng YOLOv8n-cls pretrained model")
            self.model = YOLO('yolov8n-cls.pt')  # Nano classification model
    
    def prepare_dataset_for_yolo(self, dataset_path: str, output_path: str = None) -> str:
        """
        Chuẩn bị dataset theo format YOLO classification
        
        YOLO classification format:
        dataset/
        ├── train/
        │   ├── fire/
        │   └── no_fire/
        ├── val/
        │   ├── fire/
        │   └── no_fire/
        └── test/
            ├── fire/
            └── no_fire/
        
        Args:
            dataset_path: Đường dẫn dataset gốc
            output_path: Đường dẫn output, nếu None sẽ tạo tự động
        
        Returns:
            Đường dẫn đến dataset đã chuẩn bị
        """
        if output_path is None:
            output_path = os.path.join(os.path.dirname(dataset_path), "dataset_yolo")
        
        print(f"📁 Chuẩn bị dataset YOLO tại: {output_path}")
        
        # Kiểm tra xem dataset đã có images/fire và images/no_fire chưa
        for split in ['train', 'val', 'test']:
            src_fire = os.path.join(dataset_path, split, 'images', 'fire')
            src_no_fire = os.path.join(dataset_path, split, 'images', 'no_fire')
            
            if not os.path.exists(src_fire) or not os.path.exists(src_no_fire):
                print(f"⚠️ Không tìm thấy {split}/images/fire hoặc {split}/images/no_fire")
                continue
            
            # Tạo thư mục đích
            dst_fire = os.path.join(output_path, split, 'fire')
            dst_no_fire = os.path.join(output_path, split, 'no_fire')
            
            os.makedirs(dst_fire, exist_ok=True)
            os.makedirs(dst_no_fire, exist_ok=True)
            
            # Copy hoặc symlink files
            print(f"📋 Xử lý {split}...")
            
            # Copy fire images
            fire_images = [f for f in os.listdir(src_fire) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img in fire_images:
                src = os.path.join(src_fire, img)
                dst = os.path.join(dst_fire, img)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
            
            # Copy no_fire images
            no_fire_images = [f for f in os.listdir(src_no_fire) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img in no_fire_images:
                src = os.path.join(src_no_fire, img)
                dst = os.path.join(dst_no_fire, img)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
            
            print(f"   ✅ {split}: {len(fire_images)} fire, {len(no_fire_images)} no_fire")
        
        print(f"✅ Dataset đã được chuẩn bị tại: {output_path}")
        return output_path
    
    def train(self, dataset_path: str, epochs: int = 50, imgsz: int = 224, 
              batch: int = 16, patience: int = 10, save_dir: str = "trained_models/yolo"):
        """
        Train YOLO model trên dataset
        
        Args:
            dataset_path: Đường dẫn đến dataset (format YOLO)
            epochs: Số epochs training
            imgsz: Kích thước ảnh input
            batch: Batch size
            patience: Early stopping patience
            save_dir: Thư mục lưu model
        
        Returns:
            Training results
        """
        print("🔥 Bắt đầu training YOLO model...")
        print(f"📊 Epochs: {epochs}, Image Size: {imgsz}, Batch: {batch}")
        
        # Chuẩn bị dataset nếu cần
        if not self._check_yolo_format(dataset_path):
            print("⚠️ Dataset chưa đúng format YOLO, đang chuẩn bị...")
            dataset_path = self.prepare_dataset_for_yolo(dataset_path)
        
        # Training
        results = self.model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            project=save_dir,
            name=f"fire_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            save=True,
            save_period=10,  # Lưu checkpoint mỗi 10 epochs
            device='cpu',  # Hoặc 'cuda' nếu có GPU
            verbose=True,
            plots=True
        )
        
        self.is_trained = True
        self.model_path = results.save_dir / 'weights' / 'best.pt'
        
        print(f"✅ Training hoàn thành!")
        print(f"💾 Model được lưu tại: {self.model_path}")
        
        return results
    
    def _check_yolo_format(self, dataset_path: str) -> bool:
        """Kiểm tra xem dataset có đúng format YOLO không"""
        required_dirs = [
            os.path.join(dataset_path, 'train', 'fire'),
            os.path.join(dataset_path, 'train', 'no_fire'),
            os.path.join(dataset_path, 'val', 'fire'),
            os.path.join(dataset_path, 'val', 'no_fire')
        ]
        
        return all(os.path.exists(d) for d in required_dirs)
    
    def predict(self, image_path: str, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Dự đoán cho một ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh
            conf_threshold: Ngưỡng confidence
        
        Returns:
            Dictionary chứa kết quả prediction
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
        
        # Predict
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            verbose=False
        )[0]
        
        # Parse results
        probs = results.probs
        class_names = results.names
        
        # Get prediction
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        
        # Xác định class (giả sử class 0 là fire, class 1 là no_fire hoặc ngược lại)
        pred_class = class_names[top1_idx]
        
        # Tính probability cho mỗi class
        all_probs = probs.data.cpu().numpy()
        
        # Map class names
        fire_idx = 0 if 'fire' in class_names[0].lower() else 1
        no_fire_idx = 1 - fire_idx
        
        prediction_result = {
            'prediction': 'FIRE' if top1_idx == fire_idx else 'NO FIRE',
            'confidence': float(top1_conf),
            'probability_fire': float(all_probs[fire_idx]),
            'probability_no_fire': float(all_probs[no_fire_idx]),
            'class_name': pred_class,
            'all_probabilities': {class_names[i]: float(all_probs[i]) for i in range(len(all_probs))}
        }
        
        return prediction_result
    
    def evaluate(self, dataset_path: str, split: str = 'test') -> Dict[str, Any]:
        """
        Đánh giá model trên test set
        
        Args:
            dataset_path: Đường dẫn đến dataset
            split: 'test' hoặc 'val'
        
        Returns:
            Evaluation metrics
        """
        print(f"📊 Đánh giá YOLO model trên {split} set...")
        
        # Validate
        results = self.model.val(
            data=dataset_path,
            split=split,
            verbose=True
        )
        
        # Extract metrics
        metrics = {
            'top1_accuracy': float(results.top1),
            'top5_accuracy': float(results.top5),
        }
        
        print(f"✅ Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
        print(f"✅ Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
        
        return metrics
    
    def compare_with_ml_models(self, image_paths: List[str], ml_trainer=None) -> Dict[str, Any]:
        """
        So sánh YOLO với các mô hình ML truyền thống
        
        Args:
            image_paths: Danh sách đường dẫn ảnh test
            ml_trainer: MLModelTrainer instance
        
        Returns:
            Comparison results
        """
        print("🔍 So sánh YOLO với các mô hình ML truyền thống...")
        
        yolo_predictions = []
        ml_predictions = {name: [] for name in ml_trainer.trained_models.keys()}
        ground_truth = []
        
        for img_path in image_paths:
            # YOLO prediction
            yolo_pred = self.predict(img_path)
            yolo_predictions.append(1 if yolo_pred['prediction'] == 'FIRE' else 0)
            
            # ML predictions
            if ml_trainer:
                ml_preds = ml_trainer.predict_single_image(img_path)
                for model_name, pred in ml_preds.items():
                    ml_predictions[model_name].append(1 if pred['prediction'] == 'FIRE' else 0)
            
            # Ground truth (từ tên thư mục)
            if 'fire' in img_path.lower() and 'no_fire' not in img_path.lower():
                ground_truth.append(1)
            else:
                ground_truth.append(0)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        comparison = {
            'YOLO': {
                'accuracy': accuracy_score(ground_truth, yolo_predictions),
                'precision': precision_score(ground_truth, yolo_predictions),
                'recall': recall_score(ground_truth, yolo_predictions),
                'f1_score': f1_score(ground_truth, yolo_predictions)
            }
        }
        
        if ml_trainer:
            for model_name, preds in ml_predictions.items():
                comparison[model_name] = {
                    'accuracy': accuracy_score(ground_truth, preds),
                    'precision': precision_score(ground_truth, preds),
                    'recall': recall_score(ground_truth, preds),
                    'f1_score': f1_score(ground_truth, preds)
                }
        
        return comparison
    
    def get_model_info(self) -> Dict[str, Any]:
        """Lấy thông tin về model"""
        return {
            'model_type': 'YOLOv8-Classification',
            'model_path': str(self.model_path) if self.model_path else 'pretrained',
            'is_trained': self.is_trained,
            'parameters': sum(p.numel() for p in self.model.model.parameters()) if self.model else 0
        }


def main():
    """Demo YOLO training và prediction"""
    print("🔥 YOLO Fire Detection - Demo")
    print("=" * 50)
    
    # Khởi tạo YOLO
    yolo = YOLOFireDetector()
    
    # Chuẩn bị dataset
    dataset_path = "../dataset"
    if os.path.exists(dataset_path):
        yolo_dataset = yolo.prepare_dataset_for_yolo(dataset_path)
        
        # Training (uncomment để train)
        # results = yolo.train(
        #     dataset_path=yolo_dataset,
        #     epochs=50,
        #     imgsz=224,
        #     batch=16
        # )
        
        print("\n✅ Demo hoàn thành!")
        print("📝 Để train YOLO, uncomment dòng training trong main()")
    else:
        print(f"❌ Không tìm thấy dataset: {dataset_path}")


if __name__ == "__main__":
    main()

