#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script chính để training và đánh giá các mô hình ML cho Fire Detection
"""

import os
import sys
import io

# Fix lỗi UnicodeEncodeError trên Windows console (cp1252 không hỗ trợ emoji)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import các module tự tạo
from fire_feature_extractor import DatasetLoader
from ml_models import MLModelTrainer

def setup_environment():
    """Thiết lập môi trường"""
    print("🚀 Thiết lập môi trường...")
    
    # Tạo các thư mục cần thiết
    directories = ['trained_models', 'results', 'plots', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Tạo thư mục: {directory}")
    
    # Thiết lập matplotlib
    plt.style.use('default')
    sns.set_palette("husl")
    
    print("✅ Môi trường đã sẵn sàng!")

def load_and_prepare_data(dataset_path: str, max_samples: int = None):
    """Load và chuẩn bị dữ liệu"""
    print(f"\n📁 Loading dataset từ: {dataset_path}")
    
    # Kiểm tra dataset path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path không tồn tại: {dataset_path}")
    
    # Load dataset
    loader = DatasetLoader(dataset_path)
    X, y, image_paths = loader.load_dataset(max_samples=max_samples)
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"🎯 Labels distribution: {np.bincount(y)}")
    
    return X, y, image_paths

def train_models(X: np.ndarray, y: np.ndarray, use_grid_search: bool = True):
    """Training tất cả các mô hình"""
    print(f"\n🔥 Bắt đầu training các mô hình...")
    
    # Khởi tạo trainer
    trainer = MLModelTrainer()
    
    # Training tất cả mô hình
    X_test, y_test = trainer.train_all_models(X, y, use_grid_search=use_grid_search)
    
    return trainer, X_test, y_test

def evaluate_and_compare(trainer: MLModelTrainer, X_test: np.ndarray, y_test: np.ndarray):
    """Đánh giá và so sánh các mô hình"""
    print(f"\n📊 Đánh giá và so sánh các mô hình...")
    
    # So sánh các mô hình
    comparison_df = trainer.compare_models()
    
    # Vẽ biểu đồ so sánh
    print("\n📈 Vẽ biểu đồ so sánh...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot results
    results_plot_path = f"plots/model_comparison_{timestamp}.png"
    trainer.plot_results(save_path=results_plot_path)
    
    # Plot confusion matrices
    cm_plot_path = f"plots/confusion_matrices_{timestamp}.png"
    trainer.plot_confusion_matrices(save_path=cm_plot_path)
    
    # Plot ROC curves
    roc_plot_path = f"plots/roc_curves_{timestamp}.png"
    trainer.plot_roc_curves(X_test, y_test, save_path=roc_plot_path)
    
    return comparison_df

def save_results(trainer: MLModelTrainer, comparison_df: pd.DataFrame):
    """Lưu kết quả training"""
    print(f"\n💾 Lưu kết quả...")
    
    # Lưu models
    trainer.save_models()
    
    # Lưu comparison table
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = f"results/model_comparison_{timestamp}.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"📊 Bảng so sánh đã được lưu: {comparison_path}")
    
    # Lưu summary
    summary_path = f"results/training_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("🔥 FIRE DETECTION - ML MODELS TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Models: {len(trainer.results)}\n\n")
        
        f.write("MODEL COMPARISON:\n")
        f.write("-" * 30 + "\n")
        f.write(comparison_df.to_string(index=False, float_format='%.4f'))
        f.write("\n\n")
        
        f.write("BEST MODEL:\n")
        f.write("-" * 30 + "\n")
        best_model = comparison_df.iloc[0]
        f.write(f"Model: {best_model['Model']}\n")
        f.write(f"F1-Score: {best_model['F1-Score']:.4f}\n")
        f.write(f"Accuracy: {best_model['Accuracy']:.4f}\n")
        f.write(f"Precision: {best_model['Precision']:.4f}\n")
        f.write(f"Recall: {best_model['Recall']:.4f}\n")
        f.write(f"ROC AUC: {best_model['ROC AUC']:.4f}\n")
    
    print(f"📝 Summary đã được lưu: {summary_path}")

def test_single_image(trainer: MLModelTrainer, image_path: str):
    """Test một ảnh với tất cả mô hình"""
    print(f"\n🔍 Testing ảnh: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy ảnh: {image_path}")
        return
    
    try:
        # Dự đoán với tất cả mô hình
        predictions = trainer.predict_single_image(image_path)
        
        print("\n📊 Kết quả dự đoán:")
        print("-" * 50)
        
        for model_name, pred in predictions.items():
            print(f"\n{model_name}:")
            print(f"  Prediction: {pred['prediction']}")
            print(f"  Confidence: {pred['confidence']:.3f}")
            print(f"  P(Fire): {pred['probability_fire']:.3f}")
            print(f"  P(No Fire): {pred['probability_no_fire']:.3f}")
        
        # Tìm mô hình có confidence cao nhất
        best_model = max(predictions.items(), key=lambda x: x[1]['confidence'])
        print(f"\n🏆 Mô hình tốt nhất: {best_model[0]} (Confidence: {best_model[1]['confidence']:.3f})")
        
    except Exception as e:
        print(f"❌ Lỗi khi test ảnh: {e}")

def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(description='Training và đánh giá các mô hình ML cho Fire Detection')
    parser.add_argument('--dataset', type=str, default='../dataset', 
                       help='Đường dẫn đến dataset')
    parser.add_argument('--max-samples', type=int, default=3000,
                       help='Số lượng mẫu tối đa để training (default: 3000)')
    parser.add_argument('--no-grid-search', action='store_true',
                       help='Không sử dụng Grid Search (chỉ dùng default parameters)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Tỷ lệ test set (default: 0.2)')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Số fold cho cross-validation (default: 5)')
    parser.add_argument('--test-image', type=str, default=None,
                       help='Đường dẫn đến ảnh để test')
    parser.add_argument('--load-models', type=str, default=None,
                       help='Load models từ timestamp')
    
    args = parser.parse_args()
    
    print("🔥 FIRE DETECTION - ML MODELS TRAINING & EVALUATION")
    print("=" * 60)
    
    # Thiết lập môi trường
    setup_environment()
    
    # Khởi tạo trainer
    trainer = MLModelTrainer()
    
    # Load models nếu có
    if args.load_models:
        print(f"\n📂 Loading models từ timestamp: {args.load_models}")
        trainer.load_models(args.load_models)
        
        if args.test_image:
            test_single_image(trainer, args.test_image)
        return
    
    # Load và chuẩn bị dữ liệu
    X, y, image_paths = load_and_prepare_data(args.dataset, args.max_samples)
    
    # Training các mô hình
    use_grid_search = not args.no_grid_search
    trainer, X_test, y_test = train_models(X, y, use_grid_search)
    
    # Đánh giá và so sánh
    comparison_df = evaluate_and_compare(trainer, X_test, y_test)
    
    # Lưu kết quả
    save_results(trainer, comparison_df)
    
    # Test ảnh nếu có
    if args.test_image:
        test_single_image(trainer, args.test_image)
    
    print(f"\n✅ Hoàn thành training và đánh giá!")
    print(f"📊 Kết quả được lưu trong thư mục: results/")
    print(f"📈 Biểu đồ được lưu trong thư mục: plots/")
    print(f"💾 Models được lưu trong thư mục: trained_models/")

if __name__ == "__main__":
    main() 