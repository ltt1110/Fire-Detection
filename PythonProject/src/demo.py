#!/usr/bin/env python3
"""
Demo script cho hệ thống ML Fire Detection
"""

import os
import sys
import numpy as np
from fire_feature_extractor import FireFeatureExtractor, DatasetLoader
from ml_models import MLModelTrainer

def demo_feature_extraction():
    """Demo trích xuất đặc trưng"""
    print("🔍 Demo: Trích xuất đặc trưng từ ảnh")
    print("=" * 50)
    
    # Kiểm tra dataset
    dataset_path = "../dataset"
    if not os.path.exists(dataset_path):
        print(f"❌ Không tìm thấy dataset tại: {dataset_path}")
        return False
    
    # Tìm một ảnh để test
    test_image = None
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_image = os.path.join(root, file)
                break
        if test_image:
            break
    
    if not test_image:
        print("❌ Không tìm thấy ảnh nào trong dataset")
        return False
    
    print(f"📸 Test với ảnh: {os.path.basename(test_image)}")
    
    # Trích xuất đặc trưng
    feature_extractor = FireFeatureExtractor()
    
    try:
        features = feature_extractor.extract_all_features(test_image)
        vector = feature_extractor.create_feature_vector(features)
        
        print(f"✅ Trích xuất thành công!")
        print(f"📊 Kích thước vector: {vector.shape}")
        print(f"🎨 Color histogram: {len(features['color_histogram'])} features")
        print(f"🔥 Fire features: {len(features['fire_features'])} features")
        print(f"🧱 Texture features: {len(features['texture_features'])} features")
        print(f"📈 Statistical features: {len(features['statistical_features'])} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi trích xuất đặc trưng: {e}")
        return False

def demo_dataset_loading():
    """Demo load dataset"""
    print("\n📁 Demo: Load dataset")
    print("=" * 50)
    
    dataset_path = "../dataset"
    if not os.path.exists(dataset_path):
        print(f"❌ Không tìm thấy dataset tại: {dataset_path}")
        return False, None, None, None
    
    try:
        # Load dataset với số lượng mẫu nhỏ
        loader = DatasetLoader(dataset_path)
        X, y, image_paths = loader.load_dataset(max_samples=100)
        
        print(f"✅ Load dataset thành công!")
        print(f"📊 Kích thước: {X.shape}")
        print(f"🎯 Labels: {np.bincount(y)}")
        print(f"📁 Số ảnh: {len(image_paths)}")
        
        return True, X, y, image_paths
        
    except Exception as e:
        print(f"❌ Lỗi khi load dataset: {e}")
        return False, None, None, None

def demo_model_training(X, y):
    """Demo training models"""
    print("\n🔥 Demo: Training models")
    print("=" * 50)
    
    if X is None or y is None:
        print("❌ Không có dữ liệu để training")
        return False
    
    try:
        # Khởi tạo trainer
        trainer = MLModelTrainer()
        
        # Training với Grid Search tắt để nhanh hơn
        print("🚀 Training với default parameters (không Grid Search)...")
        X_test, y_test = trainer.train_all_models(X, y, use_grid_search=False)
        
        print(f"✅ Training hoàn thành!")
        print(f"📊 Số models đã train: {len(trainer.results)}")
        
        # So sánh models
        comparison_df = trainer.compare_models()
        
        # Lưu models
        trainer.save_models()
        
        return True, trainer, X_test, y_test
        
    except Exception as e:
        print(f"❌ Lỗi khi training: {e}")
        return False, None, None, None

def demo_prediction(trainer, test_image_path):
    """Demo dự đoán"""
    print("\n🔮 Demo: Dự đoán ảnh mới")
    print("=" * 50)
    
    if trainer is None:
        print("❌ Chưa có models để dự đoán")
        return False
    
    if not os.path.exists(test_image_path):
        print(f"❌ Không tìm thấy ảnh test: {test_image_path}")
        return False
    
    try:
        # Dự đoán
        predictions = trainer.predict_single_image(test_image_path)
        
        print(f"📸 Test ảnh: {os.path.basename(test_image_path)}")
        print("\n📊 Kết quả dự đoán:")
        print("-" * 40)
        
        for model_name, pred in predictions.items():
            print(f"\n{model_name}:")
            print(f"  Prediction: {pred['prediction']}")
            print(f"  Confidence: {pred['confidence']:.3f}")
            print(f"  P(Fire): {pred['probability_fire']:.3f}")
            print(f"  P(No Fire): {pred['probability_no_fire']:.3f}")
        
        # Tìm model tốt nhất
        best_model = max(predictions.items(), key=lambda x: x[1]['confidence'])
        print(f"\n🏆 Mô hình tốt nhất: {best_model[0]} (Confidence: {best_model[1]['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi dự đoán: {e}")
        return False

def main():
    """Hàm chính"""
    print("🔥 Fire Detection ML System - Demo")
    print("=" * 60)
    
    # Demo 1: Trích xuất đặc trưng
    if not demo_feature_extraction():
        print("❌ Demo trích xuất đặc trưng thất bại")
        return
    
    # Demo 2: Load dataset
    success, X, y, image_paths = demo_dataset_loading()
    if not success:
        print("❌ Demo load dataset thất bại")
        return
    
    # Demo 3: Training models
    success, trainer, X_test, y_test = demo_model_training(X, y)
    if not success:
        print("❌ Demo training thất bại")
        return
    
    # Demo 4: Dự đoán
    if image_paths:
        test_image = image_paths[0]  # Sử dụng ảnh đầu tiên làm test
        demo_prediction(trainer, test_image)
    
    print("\n🎉 Demo hoàn thành!")
    print("📝 Để sử dụng đầy đủ:")
    print("   python run_ml_system.py --train --max-samples 1000")
    print("   python run_ml_system.py --web")

if __name__ == "__main__":
    main() 