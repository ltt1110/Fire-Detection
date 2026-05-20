#!/usr/bin/env python3
"""
Script so sánh YOLO với các mô hình ML truyền thống
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from ml_models import MLModelTrainer
from yolo_model import YOLOFireDetector, YOLO_AVAILABLE


def load_test_images(dataset_path: str, max_samples: int = 100):
    """Load test images và labels"""
    print(f"📁 Loading test images từ: {dataset_path}")
    
    image_paths = []
    labels = []
    
    # Load từ test folder
    test_fire = os.path.join(dataset_path, 'test', 'images', 'fire')
    test_no_fire = os.path.join(dataset_path, 'test', 'images', 'no_fire')
    
    # Fire images
    if os.path.exists(test_fire):
        fire_images = [os.path.join(test_fire, f) for f in os.listdir(test_fire) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:max_samples//2]
        image_paths.extend(fire_images)
        labels.extend([1] * len(fire_images))
        print(f"✅ Fire images: {len(fire_images)}")
    
    # No fire images
    if os.path.exists(test_no_fire):
        no_fire_images = [os.path.join(test_no_fire, f) for f in os.listdir(test_no_fire) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:max_samples//2]
        image_paths.extend(no_fire_images)
        labels.extend([0] * len(no_fire_images))
        print(f"✅ No fire images: {len(no_fire_images)}")
    
    print(f"📊 Total: {len(image_paths)} images")
    return image_paths, labels


def compare_models(ml_trainer: MLModelTrainer, yolo_detector: YOLOFireDetector,
                  image_paths: list, ground_truth: list):
    """So sánh tất cả các models"""
    print("\n🔍 Bắt đầu so sánh models...")
    print("=" * 60)
    
    # Collect predictions
    predictions = {}
    
    # ML Models predictions
    print("\n📊 ML Models predictions...")
    for model_name in ml_trainer.trained_models.keys():
        predictions[model_name] = []
    
    for i, img_path in enumerate(image_paths):
        try:
            ml_preds = ml_trainer.predict_single_image(img_path)
            for model_name, pred in ml_preds.items():
                pred_label = 1 if pred['prediction'] == 'FIRE' else 0
                predictions[model_name].append(pred_label)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(image_paths)} images")
        except Exception as e:
            print(f"  ⚠️ Error with {img_path}: {e}")
            for model_name in ml_trainer.trained_models.keys():
                predictions[model_name].append(0)  # Default to no fire
    
    # YOLO predictions
    if yolo_detector and yolo_detector.is_trained:
        print("\n🔥 YOLO predictions...")
        predictions['YOLO'] = []
        
        for i, img_path in enumerate(image_paths):
            try:
                yolo_pred = yolo_detector.predict(img_path)
                pred_label = 1 if yolo_pred['prediction'] == 'FIRE' else 0
                predictions['YOLO'].append(pred_label)
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(image_paths)} images")
            except Exception as e:
                print(f"  ⚠️ Error with {img_path}: {e}")
                predictions['YOLO'].append(0)
    
    # Calculate metrics
    print("\n📈 Calculating metrics...")
    results = []
    
    for model_name, preds in predictions.items():
        if len(preds) == len(ground_truth):
            metrics = {
                'Model': model_name,
                'Accuracy': accuracy_score(ground_truth, preds),
                'Precision': precision_score(ground_truth, preds, zero_division=0),
                'Recall': recall_score(ground_truth, preds, zero_division=0),
                'F1-Score': f1_score(ground_truth, preds, zero_division=0)
            }
            results.append(metrics)
    
    return pd.DataFrame(results)


def plot_comparison(comparison_df: pd.DataFrame, save_path: str = None):
    """Vẽ biểu đồ so sánh"""
    print("\n📊 Vẽ biểu đồ so sánh...")
    
    # Sắp xếp theo F1-Score
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
    
    # Tạo figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🔥 So sánh YOLO vs ML Models Truyền Thống', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    
    # Highlight YOLO với màu khác
    bar_colors = []
    for model in comparison_df['Model']:
        if model == 'YOLO':
            bar_colors.append('#FF4757')  # Đỏ tươi cho YOLO
        else:
            bar_colors.append('#778BEB')  # Xanh cho ML models
    
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        
        bars = axes[row, col].barh(comparison_df['Model'], 
                                    comparison_df[metric],
                                    color=bar_colors)
        
        axes[row, col].set_title(f'{metric}', fontsize=12, fontweight='bold')
        axes[row, col].set_xlabel('Score', fontsize=10)
        axes[row, col].set_xlim(0, 1)
        axes[row, col].grid(axis='x', alpha=0.3)
        
        # Thêm giá trị
        for j, (bar, value) in enumerate(zip(bars, comparison_df[metric])):
            axes[row, col].text(value + 0.02, bar.get_y() + bar.get_height()/2,
                              f'{value:.3f}', 
                              va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Biểu đồ đã lưu: {save_path}")
    
    plt.show()


def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(description='So sánh YOLO vs ML Models')
    parser.add_argument('--dataset', type=str, default='../dataset',
                       help='Đường dẫn đến dataset')
    parser.add_argument('--ml-timestamp', type=str, default='20250727_151800',
                       help='Timestamp của ML models')
    parser.add_argument('--yolo-model', type=str, default=None,
                       help='Đường dẫn đến YOLO model')
    parser.add_argument('--max-samples', type=int, default=100,
                       help='Số lượng ảnh test tối đa')
    parser.add_argument('--output', type=str, default='results/comparison.png',
                       help='Đường dẫn lưu biểu đồ')
    
    args = parser.parse_args()
    
    print("🔥 YOLO vs ML Models - Comparison")
    print("=" * 60)
    
    # Kiểm tra YOLO
    if not YOLO_AVAILABLE:
        print("❌ YOLO chưa được cài đặt!")
        print("📦 Cài đặt: pip install ultralytics")
        return
    
    # Load ML models
    print("\n📂 Loading ML models...")
    ml_trainer = MLModelTrainer()
    if not ml_trainer.load_models(args.ml_timestamp):
        print("❌ Không thể load ML models")
        return
    
    print(f"✅ Đã load {len(ml_trainer.trained_models)} ML models")
    
    # Load YOLO model
    yolo_detector = None
    if args.yolo_model and os.path.exists(args.yolo_model):
        print(f"\n📂 Loading YOLO model từ: {args.yolo_model}")
        yolo_detector = YOLOFireDetector(model_path=args.yolo_model)
        print("✅ YOLO model đã được load")
    else:
        print("\n⚠️ Không có YOLO model, chỉ so sánh ML models")
    
    # Load test images
    image_paths, labels = load_test_images(args.dataset, args.max_samples)
    
    if len(image_paths) == 0:
        print("❌ Không tìm thấy test images")
        return
    
    # Compare models
    comparison_df = compare_models(ml_trainer, yolo_detector, image_paths, labels)
    
    # Print results
    print("\n📊 KẾT QUẢ SO SÁNH:")
    print("=" * 80)
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Find best model
    best_model = comparison_df.iloc[0]
    print(f"\n🏆 Model tốt nhất: {best_model['Model']}")
    print(f"   F1-Score: {best_model['F1-Score']:.4f}")
    print(f"   Accuracy: {best_model['Accuracy']:.4f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/model_comparison_yolo_vs_ml.csv'
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n💾 Kết quả đã lưu: {csv_path}")
    
    # Plot comparison
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_comparison(comparison_df, args.output)
    
    print("\n✅ Hoàn thành!")


if __name__ == "__main__":
    main()

