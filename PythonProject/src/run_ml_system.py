#!/usr/bin/env python3
"""
Script tổng hợp để chạy hệ thống ML Fire Detection
"""

import os
import sys
import argparse
import subprocess
import time
import requests
from pathlib import Path

def check_port(port):
    """Kiểm tra port có đang được sử dụng không"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_web_app():
    """Khởi động web application"""
    print("🚀 Khởi động ML Web Application...")
    
    if check_port(8085):
        print("⚠️  Port 8085 đang được sử dụng")
        return False
    
    try:
        # Chạy web app trong background
        process = subprocess.Popen([
            sys.executable, "ml_web_app.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Chờ một chút để app khởi động
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ Web app đã khởi động thành công!")
            print("🌐 Truy cập: http://localhost:8085")
            return True
        else:
            print("❌ Không thể khởi động web app")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi khi khởi động web app: {e}")
        return False

def train_models(max_samples=None, use_grid_search=True, dataset_path="../dataset"):
    """Training các mô hình ML"""
    print("🔥 Bắt đầu training các mô hình ML...")
    
    cmd = [sys.executable, "train_and_evaluate.py", "--dataset", dataset_path]
    
    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])
    
    if not use_grid_search:
        cmd.append("--no-grid-search")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Training hoàn thành thành công!")
            print(result.stdout)
            return True
        else:
            print("❌ Lỗi khi training:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Lỗi khi chạy training: {e}")
        return False

def test_single_image(image_path, load_models=None):
    """Test một ảnh"""
    print(f"🔍 Testing ảnh: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy ảnh: {image_path}")
        return False
    
    cmd = [sys.executable, "train_and_evaluate.py", "--test-image", image_path]
    
    if load_models:
        cmd.extend(["--load-models", load_models])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Test hoàn thành!")
            print(result.stdout)
            return True
        else:
            print("❌ Lỗi khi test:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Lỗi khi chạy test: {e}")
        return False

def show_status():
    """Hiển thị trạng thái hệ thống"""
    print("📊 Trạng thái hệ thống ML Fire Detection")
    print("=" * 50)
    
    # Kiểm tra web app
    web_status = "🟢 Đang chạy" if check_port(8085) else "🔴 Không chạy"
    print(f"Web Application (Port 8085): {web_status}")
    
    # Kiểm tra models
    models_dir = "trained_models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and not f.startswith('scaler')]
        if model_files:
            print(f"📁 Models đã train: {len(model_files)} models")
            timestamps = list(set([f.split('_')[-1].replace('.pkl', '') for f in model_files]))
            print(f"📅 Timestamps có sẵn: {timestamps}")
        else:
            print("📁 Models: Chưa có models nào")
    else:
        print("📁 Models: Thư mục trained_models không tồn tại")
    
    # Kiểm tra results
    results_dir = "results"
    if os.path.exists(results_dir):
        result_files = os.listdir(results_dir)
        print(f"📊 Kết quả: {len(result_files)} files")
    else:
        print("📊 Kết quả: Chưa có kết quả nào")
    
    # Kiểm tra plots
    plots_dir = "plots"
    if os.path.exists(plots_dir):
        plot_files = os.listdir(plots_dir)
        print(f"📈 Biểu đồ: {len(plot_files)} files")
    else:
        print("📈 Biểu đồ: Chưa có biểu đồ nào")

def show_help():
    """Hiển thị hướng dẫn sử dụng"""
    print("🔥 ML Fire Detection System - Hướng dẫn sử dụng")
    print("=" * 60)
    print()
    print("📋 Các lệnh có sẵn:")
    print()
    print("1. Training Models:")
    print("   python run_ml_system.py --train")
    print("   python run_ml_system.py --train --max-samples 1000")
    print("   python run_ml_system.py --train --no-grid-search")
    print()
    print("2. Web Application:")
    print("   python run_ml_system.py --web")
    print()
    print("3. Test ảnh:")
    print("   python run_ml_system.py --test-image path/to/image.jpg")
    print("   python run_ml_system.py --test-image image.jpg --load-models TIMESTAMP")
    print()
    print("4. Kiểm tra trạng thái:")
    print("   python run_ml_system.py --status")
    print()
    print("5. Training + Web App:")
    print("   python run_ml_system.py --train --web")
    print()
    print("📝 Ví dụ:")
    print("   # Training nhanh với 500 mẫu")
    print("   python run_ml_system.py --train --max-samples 500 --no-grid-search")
    print()
    print("   # Training đầy đủ rồi chạy web app")
    print("   python run_ml_system.py --train --max-samples 2000 --web")
    print()
    print("   # Test ảnh với models đã train")
    print("   python run_ml_system.py --test-image ../dataset/train/images/train_1.jpg")

def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(description='Hệ thống ML Fire Detection')
    parser.add_argument('--train', action='store_true', help='Training models')
    parser.add_argument('--web', action='store_true', help='Khởi động web app')
    parser.add_argument('--test-image', type=str, help='Test một ảnh')
    parser.add_argument('--load-models', type=str, help='Load models từ timestamp')
    parser.add_argument('--max-samples', type=int, help='Số lượng mẫu tối đa để training')
    parser.add_argument('--no-grid-search', action='store_true', help='Không sử dụng Grid Search')
    parser.add_argument('--dataset', type=str, default='../dataset', help='Đường dẫn dataset')
    parser.add_argument('--status', action='store_true', help='Hiển thị trạng thái hệ thống')
    parser.add_argument('--help-cmd', action='store_true', help='Hiển thị hướng dẫn chi tiết')
    
    args = parser.parse_args()
    
    # Hiển thị help nếu không có argument nào
    if len(sys.argv) == 1:
        show_help()
        return
    
    # Hiển thị help chi tiết
    if args.help_cmd:
        show_help()
        return
    
    # Hiển thị trạng thái
    if args.status:
        show_status()
        return
    
    print("🔥 ML Fire Detection System")
    print("=" * 40)
    
    # Training models
    if args.train:
        success = train_models(
            max_samples=args.max_samples,
            use_grid_search=not args.no_grid_search,
            dataset_path=args.dataset
        )
        if not success:
            print("❌ Training thất bại!")
            return
    
    # Test ảnh
    if args.test_image:
        success = test_single_image(args.test_image, args.load_models)
        if not success:
            print("❌ Test thất bại!")
            return
    
    # Khởi động web app
    if args.web:
        success = start_web_app()
        if not success:
            print("❌ Không thể khởi động web app!")
            return
        
        print("\n🎉 Hệ thống đã sẵn sàng!")
        print("📱 Truy cập web app tại: http://localhost:8085")
        print("⏹️  Nhấn Ctrl+C để dừng")
        
        try:
            # Giữ script chạy
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")

if __name__ == "__main__":
    main() 