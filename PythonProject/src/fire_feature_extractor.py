#!/usr/bin/env python3
"""
Fire Feature Extractor for ML Training
Trích xuất đặc trưng từ ảnh để training các mô hình ML
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class FireFeatureExtractor:
    """Trích xuất đặc trưng từ ảnh cho fire detection"""
    
    def __init__(self):
        # Định nghĩa các ngưỡng màu lửa (siết chặt S>=150, V>=150)
        # Lửa thật: rất bão hoà và sáng. Màu nhạt (hoàng hôn, đèn) bị loại.
        self.fire_color_ranges = {
            # Đỏ dải chính (H: 0-10°)
            'red_lower1':    np.array([0,   150, 150]),
            'red_upper1':    np.array([10,  255, 255]),
            # Đỏ wrap-around (H: 170-180°) - dải bị bỏ sót trong code cũ!
            'red_lower2':    np.array([170, 150, 150]),
            'red_upper2':    np.array([180, 255, 255]),
            # Cam - màu chủ đạo của lửa
            'orange_lower':  np.array([10,  150, 150]),
            'orange_upper':  np.array([25,  255, 255]),
            # Vàng - chỉ lấy phần rất sáng (V>=200) để tránh đèn vàng thường
            'yellow_lower':  np.array([25,  150, 200]),
            'yellow_upper':  np.array([35,  255, 255]),
        }
    
    def preprocess_image(self, image_path: str) -> Dict[str, np.ndarray]:
        """Tiền xử lý ảnh"""
        # Load ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể load ảnh: {image_path}")
        
        # Resize về kích thước cố định
        image = cv2.resize(image, (224, 224))
        
        # Chuyển đổi màu
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return {
            'original': image,
            'rgb': rgb,
            'hsv': hsv,
            'gray': gray
        }
    
    def extract_color_histogram(self, hsv_image: np.ndarray) -> np.ndarray:
        """Trích xuất histogram màu sắc"""
        # Histogram cho từng kênh màu
        h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
        
        # Chuẩn hóa
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        
        # Kết hợp thành một vector
        color_hist = np.concatenate([h_hist, s_hist, v_hist])
        return color_hist
    
    def extract_fire_color_mask(self, hsv_image: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """Trích xuất mask màu lửa (đã cải thiện: siết ngưỡng + red wrap-around)"""
        # Tạo mask cho từng dải màu
        red_mask1  = cv2.inRange(hsv_image, self.fire_color_ranges['red_lower1'],   self.fire_color_ranges['red_upper1'])
        red_mask2  = cv2.inRange(hsv_image, self.fire_color_ranges['red_lower2'],   self.fire_color_ranges['red_upper2'])
        red_mask   = cv2.bitwise_or(red_mask1, red_mask2)   # Hợp cả 2 dải đỏ
        orange_mask = cv2.inRange(hsv_image, self.fire_color_ranges['orange_lower'], self.fire_color_ranges['orange_upper'])
        yellow_mask = cv2.inRange(hsv_image, self.fire_color_ranges['yellow_lower'], self.fire_color_ranges['yellow_upper'])
        
        # Kết hợp mask
        fire_mask = cv2.bitwise_or(red_mask, orange_mask)
        fire_mask = cv2.bitwise_or(fire_mask, yellow_mask)
        
        # Tính tỷ lệ
        total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
        red_ratio         = np.sum(red_mask > 0)    / total_pixels
        orange_ratio      = np.sum(orange_mask > 0) / total_pixels
        yellow_ratio      = np.sum(yellow_mask > 0) / total_pixels
        total_fire_ratio  = np.sum(fire_mask > 0)   / total_pixels
        
        fire_features = {
            'red_ratio':        red_ratio,
            'orange_ratio':     orange_ratio,
            'yellow_ratio':     yellow_ratio,
            'total_fire_ratio': total_fire_ratio,
            'fire_pixels':      float(np.sum(fire_mask > 0))
        }
        
        return fire_features, fire_mask
    
    def extract_texture_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Trích xuất đặc trưng texture"""
        # Gradient
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Thống kê gradient
        gradient_mean = np.mean(gradient_magnitude)
        gradient_std = np.std(gradient_magnitude)
        
        # Entropy
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Local Binary Pattern (đơn giản)
        lbp = self._compute_lbp(gray_image)
        lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()
        
        # Thống kê LBP
        lbp_mean = np.mean(lbp_hist)
        lbp_std = np.std(lbp_hist)
        
        texture_features = {
            'gradient_mean': gradient_mean,
            'gradient_std': gradient_std,
            'entropy': entropy,
            'lbp_mean': lbp_mean,
            'lbp_std': lbp_std,
            'lbp_histogram': lbp_hist
        }
        
        return texture_features
    
    def _compute_lbp(self, image: np.ndarray) -> np.ndarray:
        """
        Tính Local Binary Pattern - vectorized bằng numpy (nhanh hơn ~50x so với for loop).
        So sánh 8 neighbor pixels với center pixel, encode thành 8-bit binary.
        """
        h, w   = image.shape
        lbp    = np.zeros((h, w), dtype=np.uint8)
        center = image[1:-1, 1:-1]   # Vùng center (bỏ border 1px)

        # 8 neighbors theo thứ tự chuẩn LBP (clock-wise từ top-left)
        neighbors = [
            image[0:-2, 0:-2],   # top-left
            image[0:-2, 1:-1],   # top
            image[0:-2, 2:  ],   # top-right
            image[1:-1, 2:  ],   # right
            image[2:,   2:  ],   # bottom-right
            image[2:,   1:-1],   # bottom
            image[2:,   0:-2],   # bottom-left
            image[1:-1, 0:-2],   # left
        ]

        # Mỗi neighbor >= center đóng góp 1 bit (2^k)
        for k, neighbor in enumerate(neighbors):
            lbp[1:-1, 1:-1] += ((neighbor >= center).astype(np.uint8)) * (2 ** k)

        return lbp

    
    def extract_statistical_features(self, rgb_image: np.ndarray) -> Dict[str, float]:
        """Trích xuất đặc trưng thống kê"""
        # Thống kê từng kênh màu
        r_mean, r_std = np.mean(rgb_image[:,:,0]), np.std(rgb_image[:,:,0])
        g_mean, g_std = np.mean(rgb_image[:,:,1]), np.std(rgb_image[:,:,1])
        b_mean, b_std = np.mean(rgb_image[:,:,2]), np.std(rgb_image[:,:,2])
        
        # Tỷ lệ màu
        total_pixels = rgb_image.shape[0] * rgb_image.shape[1]
        bright_pixels = np.sum(np.mean(rgb_image, axis=2) > 200)
        dark_pixels = np.sum(np.mean(rgb_image, axis=2) < 50)
        
        bright_ratio = bright_pixels / total_pixels
        dark_ratio = dark_pixels / total_pixels
        
        # Skewness và Kurtosis
        r_skew = self._calculate_skewness(rgb_image[:,:,0])
        g_skew = self._calculate_skewness(rgb_image[:,:,1])
        b_skew = self._calculate_skewness(rgb_image[:,:,2])
        
        statistical_features = {
            'r_mean': r_mean, 'r_std': r_std, 'r_skew': r_skew,
            'g_mean': g_mean, 'g_std': g_std, 'g_skew': g_skew,
            'b_mean': b_mean, 'b_std': b_std, 'b_skew': b_skew,
            'bright_ratio': bright_ratio,
            'dark_ratio': dark_ratio
        }
        
        return statistical_features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Tính skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def extract_fire_specific_features(self, hsv_image: np.ndarray,
                                        gray_image: np.ndarray,
                                        fire_mask: np.ndarray) -> Dict[str, float]:
        """
        Trích xuất đặc trưng hình học/cạnh đặc trưng của lửa thật.
        Giúp phân biệt lửa với màu cam/đỏ thông thường (hoàng hôn, đèn, vật thể).
        """
        features = {}
        fire_pixel_count = int(np.sum(fire_mask > 0))
        total_pixels = float(fire_mask.shape[0] * fire_mask.shape[1])

        # --- Đặc trưng 1: Độ mờ cạnh vùng lửa ---
        # Lửa thật có cạnh mờ, lung linh. Vật thể cứng có cạnh sắc nét.
        if fire_pixel_count > 200:
            fire_region = cv2.bitwise_and(gray_image, gray_image, mask=fire_mask)
            laplacian = cv2.Laplacian(fire_region, cv2.CV_64F)
            laplacian_var = np.var(laplacian[fire_mask > 0])
            features['edge_blurriness'] = 1.0 / (laplacian_var + 1e-5)  # Cao = mờ = lửa
        else:
            features['edge_blurriness'] = 0.0

        # --- Đặc trưng 2: Phân bố dọc của vùng lửa ---
        # Lửa thường bốc lên: phần dưới đậm hơn phần trên
        h = fire_mask.shape[0]
        bottom_fire = float(np.sum(fire_mask[h // 2:] > 0))
        top_fire    = float(np.sum(fire_mask[:h // 2] > 0))
        denom = float(fire_pixel_count) + 1e-5
        features['bottom_fire_ratio'] = bottom_fire / denom
        features['top_fire_ratio']    = top_fire    / denom

        # --- Đặc trưng 3: Tính liên thông của vùng lửa ---
        # Lửa thường tạo thành 1 vùng lớn liên tục, không rải rác
        contours, _ = cv2.findContours(
            fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest      = max(contours, key=cv2.contourArea)
            largest_area = float(cv2.contourArea(largest))
            perimeter    = float(cv2.arcLength(largest, True)) + 1e-5
            # Compactness: hình tròn = 1, lửa nhọn = gần 0
            features['fire_compactness']       = 4.0 * np.pi * largest_area / (perimeter ** 2)
            # Tỷ lệ vùng lửa lớn nhất / tổng vùng lửa
            features['largest_contour_ratio']  = largest_area / denom
            features['num_fire_contours']      = float(len(contours))
        else:
            features['fire_compactness']       = 0.0
            features['largest_contour_ratio']  = 0.0
            features['num_fire_contours']      = 0.0

        # --- Đặc trưng 4: Độ biến thiên nội bộ trong vùng lửa ---
        # Lửa có gradient sáng-tối nội bộ cao do các lưỡi lửa
        if fire_pixel_count > 200:
            v_channel   = hsv_image[:, :, 2]
            fire_pixels = v_channel[fire_mask > 0]
            features['fire_internal_std']  = float(np.std(fire_pixels))
            features['fire_internal_mean'] = float(np.mean(fire_pixels))
        else:
            features['fire_internal_std']  = 0.0
            features['fire_internal_mean'] = 0.0

        # --- Đặc trưng 5: Tỷ lệ vùng lửa so với tổng ảnh ---
        features['fire_coverage'] = float(fire_pixel_count) / total_pixels

        return features

    def extract_all_features(self, image_path: str) -> Dict[str, Any]:
        """Trích xuất tất cả đặc trưng (đã bổ sung fire_specific_features)"""
        # Tiền xử lý
        processed = self.preprocess_image(image_path)
        
        # Trích xuất đặc trưng
        color_hist           = self.extract_color_histogram(processed['hsv'])
        fire_features, fire_mask = self.extract_fire_color_mask(processed['hsv'])
        texture_features     = self.extract_texture_features(processed['gray'])
        statistical_features = self.extract_statistical_features(processed['rgb'])
        fire_specific        = self.extract_fire_specific_features(
                                   processed['hsv'], processed['gray'], fire_mask)
        
        # Tạo vector đặc trưng
        feature_vector = {
            'color_histogram':      color_hist,
            'fire_features':        fire_features,
            'texture_features':     texture_features,
            'statistical_features': statistical_features,
            'fire_specific':        fire_specific,
            'processed_images':     processed,
            'fire_mask':            fire_mask
        }
        
        return feature_vector
    
    def create_feature_vector(self, feature_data: Dict[str, Any]) -> np.ndarray:
        """
        Tạo vector đặc trưng từ dữ liệu đã trích xuất.

        Breakdown:
          - Color Histogram (692): H(180) + S(256) + V(256)
          - Fire Color Mask  (5) : red_ratio, orange_ratio, yellow_ratio,
                                   total_fire_ratio, fire_pixels
          - Texture Features (5) : gradient_mean, gradient_std, entropy,
                                   lbp_mean, lbp_std
          - Statistical     (11) : r/g/b mean+std+skew, bright_ratio, dark_ratio
          - Fire-Specific    (9) : edge_blurriness, bottom/top_fire_ratio,
                                   fire_compactness, largest_contour_ratio,
                                   num_fire_contours, fire_internal_std/mean,
                                   fire_coverage

        Tổng: 692 + 5 + 5 + 11 + 9 = 722 features
        """
        vectors = []
        
        # 1. Color histogram (692 features: 180 H + 256 S + 256 V)
        vectors.append(feature_data['color_histogram'])
        
        # 2. Fire color mask features (5 features)
        fire_feat = list(feature_data['fire_features'].values())
        vectors.append(np.array(fire_feat, dtype=np.float64))
        
        # 3. Texture features (5 features)
        texture_feat = [
            feature_data['texture_features']['gradient_mean'],
            feature_data['texture_features']['gradient_std'],
            feature_data['texture_features']['entropy'],
            feature_data['texture_features']['lbp_mean'],
            feature_data['texture_features']['lbp_std']
        ]
        vectors.append(np.array(texture_feat, dtype=np.float64))
        
        # 4. Statistical features (11 features)
        stat_feat = list(feature_data['statistical_features'].values())
        vectors.append(np.array(stat_feat, dtype=np.float64))

        # 5. Fire-specific geometric/edge features (9 features) [MỚI]
        fire_specific_feat = [
            feature_data['fire_specific']['edge_blurriness'],
            feature_data['fire_specific']['bottom_fire_ratio'],
            feature_data['fire_specific']['top_fire_ratio'],
            feature_data['fire_specific']['fire_compactness'],
            feature_data['fire_specific']['largest_contour_ratio'],
            feature_data['fire_specific']['num_fire_contours'],
            feature_data['fire_specific']['fire_internal_std'],
            feature_data['fire_specific']['fire_internal_mean'],
            feature_data['fire_specific']['fire_coverage'],
        ]
        vectors.append(np.array(fire_specific_feat, dtype=np.float64))
        
        # Kết hợp tất cả thành một vector
        combined_vector = np.concatenate(vectors)
        return combined_vector

class DatasetLoader:
    """Load và chuẩn bị dataset cho training"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.feature_extractor = FireFeatureExtractor()
        
    def load_dataset(self, max_samples: int = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load toàn bộ dataset và trích xuất features"""
        print(f"📁 Loading dataset từ: {self.dataset_path}")
        
        # Tìm tất cả ảnh trong dataset
        image_paths = []
        labels = []
        
        # Load từ train folder
        train_images_dir = os.path.join(self.dataset_path, 'train', 'images')
        
        if os.path.exists(train_images_dir):
            # Kiểm tra cấu trúc thư mục
            train_subdirs = [d for d in os.listdir(train_images_dir) if os.path.isdir(os.path.join(train_images_dir, d))]
            print(f"📁 Train subdirectories: {train_subdirs}")
            
            for subdir in train_subdirs:
                subdir_path = os.path.join(train_images_dir, subdir)
                images = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                print(f"📁 {subdir}: {len(images)} images")
                
                # Xác định label từ tên thư mục
                label = 1 if subdir.lower() == 'fire' else 0
                
                for img_file in images:
                    img_path = os.path.join(subdir_path, img_file)
                    image_paths.append(img_path)
                    labels.append(label)
        
        # Load từ val folder
        val_images_dir = os.path.join(self.dataset_path, 'val', 'images')
        
        if os.path.exists(val_images_dir):
            val_subdirs = [d for d in os.listdir(val_images_dir) if os.path.isdir(os.path.join(val_images_dir, d))]
            print(f"📁 Val subdirectories: {val_subdirs}")
            
            for subdir in val_subdirs:
                subdir_path = os.path.join(val_images_dir, subdir)
                images = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                print(f"📁 {subdir}: {len(images)} images")
                
                label = 1 if subdir.lower() == 'fire' else 0
                
                for img_file in images:
                    img_path = os.path.join(subdir_path, img_file)
                    image_paths.append(img_path)
                    labels.append(label)
        
        # Load từ test folder
        test_images_dir = os.path.join(self.dataset_path, 'test', 'images')
        
        if os.path.exists(test_images_dir):
            test_subdirs = [d for d in os.listdir(test_images_dir) if os.path.isdir(os.path.join(test_images_dir, d))]
            print(f"📁 Test subdirectories: {test_subdirs}")
            
            for subdir in test_subdirs:
                subdir_path = os.path.join(test_images_dir, subdir)
                images = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                print(f"📁 {subdir}: {len(images)} images")
                
                label = 1 if subdir.lower() == 'fire' else 0
                
                for img_file in images:
                    img_path = os.path.join(subdir_path, img_file)
                    image_paths.append(img_path)
                    labels.append(label)
        
        print(f"📊 Tổng cộng: {len(image_paths)} ảnh")
        
        # Giới hạn số lượng mẫu nếu cần
        if max_samples and len(image_paths) > max_samples:
            # Lấy ngẫu nhiên max_samples mẫu, đảm bảo cân bằng
            fire_indices = [i for i, label in enumerate(labels) if label == 1]
            no_fire_indices = [i for i, label in enumerate(labels) if label == 0]
            
            print(f"🔍 Debug: Fire indices: {len(fire_indices)}, No fire indices: {len(no_fire_indices)}")
            
            # Đảm bảo có cả hai class
            if len(fire_indices) == 0 or len(no_fire_indices) == 0:
                print("⚠️ Cảnh báo: Chỉ có một class trong dataset!")
                if len(fire_indices) == 0:
                    # Chỉ có no_fire
                    selected_indices = np.random.choice(no_fire_indices, min(len(no_fire_indices), max_samples), replace=False)
                    image_paths = [image_paths[int(i)] for i in selected_indices]
                    labels = [labels[int(i)] for i in selected_indices]
                    print(f"📊 Giới hạn còn {len(image_paths)} mẫu (Fire: 0, No Fire: {len(image_paths)})")
                else:
                    # Chỉ có fire
                    selected_indices = np.random.choice(fire_indices, min(len(fire_indices), max_samples), replace=False)
                    image_paths = [image_paths[int(i)] for i in selected_indices]
                    labels = [labels[int(i)] for i in selected_indices]
                    print(f"📊 Giới hạn còn {len(image_paths)} mẫu (Fire: {len(image_paths)}, No Fire: 0)")
            else:
                # Có cả hai class, cân bằng
                fire_samples = min(len(fire_indices), max_samples // 2)
                no_fire_samples = min(len(no_fire_indices), max_samples // 2)
                
                # Đảm bảo có ít nhất 1 mẫu từ mỗi class
                if fire_samples == 0:
                    fire_samples = 1
                    no_fire_samples = max_samples - 1
                elif no_fire_samples == 0:
                    no_fire_samples = 1
                    fire_samples = max_samples - 1
                
                # Đảm bảo không vượt quá max_samples
                total_samples = fire_samples + no_fire_samples
                if total_samples > max_samples:
                    if fire_samples > no_fire_samples:
                        fire_samples = max_samples - no_fire_samples
                    else:
                        no_fire_samples = max_samples - fire_samples
                
                selected_fire = np.random.choice(fire_indices, fire_samples, replace=False)
                selected_no_fire = np.random.choice(no_fire_indices, no_fire_samples, replace=False)
                
                selected_indices = np.concatenate([selected_fire, selected_no_fire])
                np.random.shuffle(selected_indices)
                
                image_paths = [image_paths[int(i)] for i in selected_indices]
                labels = [labels[int(i)] for i in selected_indices]
                print(f"📊 Giới hạn còn {len(image_paths)} mẫu (Fire: {fire_samples}, No Fire: {no_fire_samples})")
        
        # Trích xuất features
        print("🔍 Trích xuất features...")
        features = []
        valid_paths = []
        valid_labels = []
        
        for i, (img_path, label) in enumerate(zip(image_paths, labels)):
            try:
                feature_vector = self.feature_extractor.extract_all_features(img_path)
                vector = self.feature_extractor.create_feature_vector(feature_vector)
                features.append(vector)
                valid_paths.append(img_path)
                valid_labels.append(label)
                
                if (i + 1) % 100 == 0:
                    print(f"📊 Đã xử lý {i + 1}/{len(image_paths)} ảnh")
                    
            except Exception as e:
                print(f"⚠️ Lỗi khi xử lý {img_path}: {e}")
                continue
        
        X = np.array(features)
        y = np.array(valid_labels)
        
        print(f"✅ Dataset đã được load thành công!")
        print(f"📊 Kích thước: {X.shape}")
        print(f"🎯 Số mẫu có lửa: {np.sum(y == 1)}")
        print(f"❌ Số mẫu không lửa: {np.sum(y == 0)}")
        
        return X, y, valid_paths 