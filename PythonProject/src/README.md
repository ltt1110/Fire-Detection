# 🔥 Fire Detection - Machine Learning System

Hệ thống Machine Learning hoàn chỉnh để phát hiện lửa từ ảnh, bao gồm training, đánh giá và deployment.

## 📊 Kết quả Training (27/07/2025)

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| **KNN** | 60% | 61% | 92% | 73% | 0.49 |
| **SVM** | 60% | 61% | 92% | 73% | 0.57 |
| **Logistic Regression** | 55% | 60% | 75% | 67% | 0.60 |
| **Random Forest** | 45% | 54% | 58% | 56% | 0.38 |
| **Decision Tree** | 40% | 50% | 50% | 50% | 0.38 |

**Model tốt nhất:** KNN và SVM (F1-Score: 73%)

## 🚀 Quick Start

### 1. Cài đặt dependencies
```bash
cd src
pip install -r requirements.txt
```

### 2. Training models
```bash
# Training nhanh với 500 mẫu
python train_and_evaluate.py --max-samples 500 --no-grid-search

# Training đầy đủ
python train_and_evaluate.py
```

### 3. Chạy web app
```bash
python ml_web_app.py
```
Truy cập: `http://localhost:8085`

## 📁 Cấu trúc dự án

```
src/
├── 🔧 Core Modules
│   ├── fire_feature_extractor.py    # Trích xuất đặc trưng từ ảnh
│   ├── ml_models.py                 # Định nghĩa và training ML models
│   ├── train_and_evaluate.py        # Pipeline training chính
│   └── ml_web_app.py               # Web application
│
├── 🌐 Web Interface
│   ├── templates/
│   │   └── ml_index.html           # Giao diện web
│   └── uploads/                    # Ảnh upload (tự tạo)
│
├── 📊 Results & Models
│   ├── trained_models/             # Models đã train
│   ├── results/                    # Kết quả đánh giá
│   └── plots/                      # Biểu đồ so sánh
│
├── 🛠️ Utilities
│   ├── balance_dataset.py          # Cân bằng dataset
│   ├── debug_dataset.py            # Debug dataset
│   └── demo.py                     # Demo nhanh
│
└── 📋 Documentation
    ├── README.md                   # Tài liệu này
    ├── SUMMARY.md                  # Tóm tắt hệ thống
    └── requirements.txt            # Dependencies
```

## 🔍 Tính năng chính

### 1. **Feature Extraction (714 features)**
- **Color Features (692)**: HSV histogram, fire color mask
- **Texture Features (5)**: Gradient, entropy, LBP
- **Statistical Features (12)**: RGB statistics, brightness

### 2. **ML Models Supported**
- **K-Nearest Neighbors (KNN)** - Đơn giản, hiệu quả
- **Support Vector Machine (SVM)** - Tốt với dữ liệu nhiều chiều
- **Decision Tree** - Dễ hiểu, có thể giải thích
- **Logistic Regression** - Nhanh, ổn định
- **Random Forest** - Hiệu quả cao, ít overfitting

### 3. **Evaluation Metrics**
- Accuracy, Precision, Recall, F1-Score
- ROC AUC, Confusion Matrix
- Cross-validation scores

### 4. **Web Interface**
- Upload ảnh drag & drop
- Real-time prediction với tất cả models
- Visual results và confidence scores
- Model comparison

## 📈 Sử dụng chi tiết

### Training Models

```bash
# Training cơ bản
python train_and_evaluate.py

# Training với giới hạn mẫu
python train_and_evaluate.py --max-samples 1000

# Training nhanh (không grid search)
python train_and_evaluate.py --no-grid-search

# Training với dataset tùy chỉnh
python train_and_evaluate.py --dataset /path/to/dataset
```

### Test ảnh đơn lẻ

```bash
# Test sau training
python train_and_evaluate.py --test-image test_image.jpg

# Test với models đã lưu
python train_and_evaluate.py --load-models 20250727_151800 --test-image test_image.jpg
```

### Web Application

```bash
# Khởi động web app
python ml_web_app.py

# Web app với port tùy chỉnh
python ml_web_app.py --port 8080
```

## 🎯 Kết quả thực tế

### Performance Analysis
- **KNN & SVM** cho kết quả tốt nhất với F1-Score 73%
- **Logistic Regression** ổn định với F1-Score 67%
- **Random Forest** và **Decision Tree** cần cải thiện hyperparameters

### Recommendations
1. **Production**: Sử dụng KNN hoặc SVM
2. **Real-time**: Logistic Regression (nhanh nhất)
3. **Interpretability**: Decision Tree (dễ giải thích)

## 🔧 Troubleshooting

### Lỗi thường gặp

**1. "No module named 'cv2'"**
```bash
pip install opencv-python
```

**2. "Dataset path không tồn tại"**
```bash
# Kiểm tra cấu trúc dataset
ls ../dataset/train/images/
```

**3. "Memory error khi training"**
```bash
# Giảm số lượng mẫu
python train_and_evaluate.py --max-samples 500
```

**4. "Port already in use"**
```bash
# Thay đổi port
python ml_web_app.py --port 8086
```

## 📊 Dataset Requirements

```
dataset/
├── train/
│   ├── images/          # Ảnh training
│   └── labels/          # Labels training
├── val/
│   ├── images/          # Ảnh validation
│   └── labels/          # Labels validation
└── test/
    ├── images/          # Ảnh test
    └── labels/          # Labels test
```

## 🚀 Deployment

### Local Development
```bash
cd src
python ml_web_app.py
```

### Production (Docker)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8085
CMD ["python", "ml_web_app.py"]
```

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Trang chủ |
| `/upload` | POST | Upload và dự đoán ảnh |
| `/health` | GET | Health check |
| `/models` | GET | Thông tin models |
| `/load-models` | POST | Load models |
| `/train-status` | GET | Trạng thái training |

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.

## 📞 Liên hệ

- **Author**: ML Fire Detection Team
- **Email**: contact@firedetection.com
- **Project**: [GitHub Repository](https://github.com/firedetection/ml-system)

---

⭐ **Star this repository nếu bạn thấy hữu ích!** 