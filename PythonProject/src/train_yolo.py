#!/usr/bin/env python3
"""
Script để train YOLO model cho Fire Detection
"""

import os
import sys
import argparse
from datetime import datetime
from yolo_model import YOLOFireDetector, YOLO_AVAILABLE


def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(description='Train YOLO model cho Fire Detection')
    parser.add_argument('--dataset', type=str, default='../dataset',
                       help='Đường dẫn đến dataset')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Số epochs training (default: 50)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--imgsz', type=int, default=224,
                       help='Kích thước ảnh input (default: 224)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--model', type=str, default=None,
                       help='Đường dẫn đến pretrained model (optional)')
    parser.add_argument('--test', action='store_true',
                       help='Test model sau khi train')
    
    args = parser.parse_args()
    
    print("🔥 YOLO Fire Detection - Training")
    print("=" * 60)
    
    # Kiểm tra YOLO có sẵn không
    if not YOLO_AVAILABLE:
        print("❌ Ultralytics YOLO chưa được cài đặt!")
        print("📦 Cài đặt: pip install ultralytics")
        return
    
    # Kiểm tra dataset
    if not os.path.exists(args.dataset):
        print(f"❌ Không tìm thấy dataset: {args.dataset}")
        return
    
    # Khởi tạo YOLO
    print(f"📦 Khởi tạo YOLO model...")
    yolo = YOLOFireDetector(model_path=args.model)
    
    # Chuẩn bị dataset
    print(f"\n📁 Chuẩn bị dataset...")
    yolo_dataset = yolo.prepare_dataset_for_yolo(args.dataset)
    
    # Training
    print(f"\n🔥 Bắt đầu training...")
    print(f"   📊 Epochs: {args.epochs}")
    print(f"   📦 Batch size: {args.batch}")
    print(f"   🖼️  Image size: {args.imgsz}")
    print(f"   ⏱️  Patience: {args.patience}")
    print()
    
    try:
        results = yolo.train(
            dataset_path=yolo_dataset,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience
        )
        
        print(f"\n✅ Training hoàn thành!")
        print(f"💾 Model được lưu tại: {yolo.model_path}")
        
        # Test nếu được yêu cầu
        if args.test:
            print(f"\n📊 Đánh giá model...")
            metrics = yolo.evaluate(yolo_dataset, split='test')
            
            print(f"\n📈 Kết quả test:")
            print(f"   Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
            print(f"   Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
        
        # Lưu thông tin model
        model_info = yolo.get_model_info()
        print(f"\n📝 Thông tin model:")
        print(f"   Type: {model_info['model_type']}")
        print(f"   Parameters: {model_info['parameters']:,}")
        print(f"   Trained: {model_info['is_trained']}")
        
    except Exception as e:
        print(f"\n❌ Lỗi khi training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n🎉 Hoàn thành!")
    print(f"📝 Để sử dụng model trong web app:")
    print(f"   python ml_web_app.py --use-yolo --yolo-model {yolo.model_path}")


if __name__ == "__main__":
    main()

