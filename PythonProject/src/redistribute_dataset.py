#!/usr/bin/env python3
import os
import shutil
import random
import uuid
from pathlib import Path


def redistribute_class(dataset_path, class_name, train_ratio=0.7, val_ratio=0.2):
    print(f"📦 Đang phân bổ lại cho nhóm: {class_name}")

    subfolders = ['train', 'val', 'test']
    all_images = []

    # 1. Thu thập tất cả ảnh
    for folder in subfolders:
        img_dir = os.path.join(dataset_path, folder, 'images', class_name)
        if os.path.exists(img_dir):
            images = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            all_images.extend(images)

    if not all_images:
        print(f"⚠️ Không tìm thấy ảnh nào cho nhóm {class_name}")
        return

    random.shuffle(all_images)
    total_count = len(all_images)

    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)

    # 2. Chuyển vào temp_dir và ĐỔI TÊN NGẮN (Khắc phục lỗi MAX_PATH của Windows)
    temp_dir = os.path.join(dataset_path, f"temp_{class_name}")
    os.makedirs(temp_dir, exist_ok=True)

    new_all_images = []
    for img_path in all_images:
        ext = os.path.splitext(img_path)[1].lower()
        # Tạo tên mới chuẩn mực: classname_8kytungaunhien.jpg
        short_filename = f"{class_name}_{uuid.uuid4().hex[:8]}{ext}"
        temp_path = os.path.join(temp_dir, short_filename)

        try:
            shutil.move(img_path, temp_path)
            new_all_images.append(temp_path)
        except Exception as e:
            # Nếu tên file quá dài đến mức Windows không thèm nhận diện, ta đành bỏ qua nó
            print(f"  ⚠️ Bỏ qua 1 file lỗi (tên quá dài hoặc không hợp lệ)")
            continue

    # Cập nhật lại số lượng chính xác sau khi lọc lỗi
    train_imgs = new_all_images[:train_count]
    val_imgs = new_all_images[train_count:train_count + val_count]
    test_imgs = new_all_images[train_count + val_count:]

    print(
        f"📊 Tổng số ảnh hợp lệ {class_name}: {len(new_all_images)} -> Chia lại: Train ({len(train_imgs)}), Val ({len(val_imgs)}), Test ({len(test_imgs)})")

    # 3. Di chuyển từ thư mục tạm vào vị trí mới
    distribution = {
        'train': train_imgs,
        'val': val_imgs,
        'test': test_imgs
    }

    for target_folder, img_list in distribution.items():
        target_dir = os.path.join(dataset_path, target_folder, 'images', class_name)
        os.makedirs(target_dir, exist_ok=True)

        for temp_path in img_list:
            filename = os.path.basename(temp_path)
            dst_final = os.path.join(target_dir, filename)
            shutil.move(temp_path, dst_final)

    # Xóa thư mục tạm
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"✅ Hoàn thành phân bổ lại cho nhóm {class_name}!\n")


def main():
    dataset_path = "../dataset"
    if not os.path.exists(dataset_path):
        print(f"❌ Không tìm thấy thư mục dataset tại: {dataset_path}")
        return

    redistribute_class(dataset_path, 'fire')
    redistribute_class(dataset_path, 'no_fire')
    print("🎯 Toàn bộ Dataset đã được làm sạch tên và đưa về tỷ lệ chuẩn!")


if __name__ == "__main__":
    main()