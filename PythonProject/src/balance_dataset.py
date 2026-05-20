#!/usr/bin/env python3
"""
Script để cân bằng dataset bằng cách xóa bớt ảnh fire
"""

import os
import shutil
import random
from pathlib import Path

def balance_dataset(dataset_path: str, target_ratio: float = 1.0):
    """
    Cân bằng dataset bằng cách xóa bớt ảnh fire
    
    Args:
        dataset_path: Đường dẫn đến dataset
        target_ratio: Tỷ lệ fire/no_fire mong muốn (1.0 = cân bằng)
    """
    print(f"🔧 Cân bằng dataset tại: {dataset_path}")
    
    # Các thư mục cần xử lý
    folders = ['train', 'val', 'test']
    
    for folder in folders:
        fire_dir = os.path.join(dataset_path, folder, 'images', 'fire')
        no_fire_dir = os.path.join(dataset_path, folder, 'images', 'no_fire')
        
        if not os.path.exists(fire_dir) or not os.path.exists(no_fire_dir):
            print(f"⚠️ Bỏ qua {folder}: không tìm thấy thư mục fire hoặc no_fire")
            continue
        
        # Đếm số ảnh
        fire_images = [f for f in os.listdir(fire_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        no_fire_images = [f for f in os.listdir(no_fire_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\n📁 {folder}:")
        print(f"  🔥 Fire images: {len(fire_images)}")
        print(f"  ❌ No fire images: {len(no_fire_images)}")
        
        # Tính số ảnh fire cần giữ lại
        target_fire_count = int(len(no_fire_images) * target_ratio)
        
        if len(fire_images) > target_fire_count:
            # Xóa bớt ảnh fire
            images_to_remove = len(fire_images) - target_fire_count
            images_to_delete = random.sample(fire_images, images_to_remove)
            
            print(f"  🗑️ Xóa {images_to_remove} ảnh fire để cân bằng...")
            
            for img in images_to_delete:
                img_path = os.path.join(fire_dir, img)
                try:
                    os.remove(img_path)
                    print(f"    ✅ Đã xóa: {img}")
                except Exception as e:
                    print(f"    ❌ Lỗi khi xóa {img}: {e}")
            
            print(f"  ✅ Hoàn thành! Còn lại {target_fire_count} ảnh fire")
        else:
            print(f"  ℹ️ Không cần xóa (fire images đã ít hơn mục tiêu)")
    
    print("\n🎯 Dataset đã được cân bằng!")

def backup_dataset(dataset_path: str):
    """Tạo backup trước khi xóa"""
    backup_path = dataset_path + "_backup"
    
    if os.path.exists(backup_path):
        print(f"⚠️ Backup đã tồn tại: {backup_path}")
        return backup_path
    
    print(f"💾 Tạo backup tại: {backup_path}")
    shutil.copytree(dataset_path, backup_path)
    print("✅ Backup hoàn thành!")
    return backup_path

def main():
    dataset_path = "../dataset"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Không tìm thấy dataset: {dataset_path}")
        return
    
    # Tạo backup
    backup_path = backup_dataset(dataset_path)
    
    # Cân bằng dataset (tỷ lệ 1:1)
    balance_dataset(dataset_path, target_ratio=1.0)
    
    print(f"\n📊 Kết quả:")
    print(f"  📁 Dataset gốc: {dataset_path}")
    print(f"  💾 Backup: {backup_path}")
    print(f"  🎯 Tỷ lệ fire/no_fire: 1:1")

if __name__ == "__main__":
    main() 