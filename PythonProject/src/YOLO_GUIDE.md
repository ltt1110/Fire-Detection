# 🔥 YOLO Fire Detection - Hướng dẫn sử dụng

## 📋 Tổng quan

Tích hợp **YOLOv8 Classification** để so sánh với các mô hình ML truyền thống (KNN, SVM, Decision Tree, Logistic Regression, Random Forest).

## 🎯 Tại sao thêm YOLO?

### **Deep Learning vs Traditional ML**

| **Đặc điểm** | **YOLO (Deep Learning)** | **ML Truyền thống** |
|-------------|--------------------------|---------------------|
| **Feature Extraction** | Tự động học từ data | Hand-crafted (714 features) |
| **Training Time** | Lâu hơn (GPU recommended) | Nhanh hơn |
| **Accuracy** | Thường cao hơn | Phụ thuộc features |
| **Model Size** | Lớn (~6-50MB) | Nhỏ (<1MB) |
| **Inference Speed** | Nhanh (GPU) | Rất nhanh (CPU) |
| **Interpretability** | Black box | Dễ giải thích hơn |

## 🚀 Cài đặt

```bash
cd src
pip install ultralytics torch torchvision
```

## 📊 Training YOLO Model

### 1. Training cơ bản

```bash
python train_yolo.py --dataset ../dataset --epochs 50
```

### 2. Training nhanh (ít epochs)

```bash
python train_yolo.py --dataset ../dataset --epochs 30 --batch 16
```

### 3. Training với các tham số tùy chỉnh

```bash
python train_yolo.py \
  --dataset ../dataset \
  --epochs 100 \
  --batch 32 \
  --imgsz 224 \
  --patience 20
```

### 4. Training và test ngay

```bash
python train_yolo.py --dataset ../dataset --epochs 50 --test
```

## 🔍 So sánh YOLO với ML Models

### Chạy script so sánh

```bash
# So sánh với ML models đã train
python compare_models.py \
  --dataset ../dataset \
  --ml-timestamp 20250727_151800 \
  --yolo-model trained_models/yolo/fire_detection_YYYYMMDD_HHMMSS/weights/best.pt \
  --max-samples 100
```

### Kết quả mong đợi

```
📊 KẾT QUẢ SO SÁNH:
================================================================================
      Model  Accuracy  Precision    Recall  F1-Score
       YOLO    0.9200     0.9100    0.9300    0.9200
        KNN    0.7300     0.7200    0.7500    0.7350
        SVM    0.7300     0.7100    0.7600    0.7345
    ...
```

## 🌐 Sử dụng trong Web App

YOLO được tích hợp tự động vào web app nếu model đã được train:

```bash
# Start web app (YOLO sẽ tự động load nếu có)
python ml_web_app.py
```

Web app sẽ hiển thị predictions từ:
- ✅ 5 ML Models truyền thống
- ✅ YOLO (nếu đã train)

## 📈 Cấu trúc Output

```
trained_models/
└── yolo/
    └── fire_detection_20250101_120000/
        ├── weights/
        │   ├── best.pt          # Model tốt nhất
        │   └── last.pt          # Model epoch cuối
        ├── results.png          # Training curves
        ├── confusion_matrix.png # Confusion matrix
        └── args.yaml           # Training arguments
```

## 🎨 Visualizations

YOLO tự động tạo các biểu đồ:

1. **Training curves**: Loss, accuracy qua các epochs
2. **Confusion matrix**: Hiệu suất trên validation set
3. **Prediction examples**: Ví dụ predictions

## ⚙️ Hyperparameters

### Recommended cho CPU

```bash
python train_yolo.py \
  --epochs 30 \
  --batch 8 \
  --imgsz 224
```

### Recommended cho GPU

```bash
python train_yolo.py \
  --epochs 100 \
  --batch 32 \
  --imgsz 640
```

## 📊 Model Architecture

**YOLOv8n-cls** (Nano Classification):
- Parameters: ~2.7M
- Size: ~6MB
- Input: 224x224 RGB
- Output: 2 classes (fire, no_fire)
- Backbone: CSPDarknet
- Speed: ~1ms/image (GPU)

## 🔧 Troubleshooting

### 1. Out of Memory

```bash
# Giảm batch size
python train_yolo.py --batch 4

# Giảm image size
python train_yolo.py --imgsz 128
```

### 2. Training quá chậm

```bash
# Dùng fewer epochs
python train_yolo.py --epochs 20

# Dùng pretrained weights
# (YOLO mặc định đã dùng pretrained)
```

### 3. Overfitting

```bash
# Tăng patience cho early stopping
python train_yolo.py --patience 20

# Data augmentation (tự động trong YOLO)
```

## 📝 Tips & Best Practices

### 1. **Dataset Balance**
- Đảm bảo số lượng fire và no_fire images cân bằng
- Sử dụng `balance_dataset.py` nếu cần

### 2. **Training Strategy**
- Bắt đầu với epochs thấp (30-50) để test
- Tăng dần nếu kết quả tốt
- Monitor validation metrics để tránh overfitting

### 3. **Hyperparameter Tuning**
- Batch size: 8-32 (tùy RAM/VRAM)
- Image size: 224-640 (càng lớn càng chính xác nhưng chậm hơn)
- Learning rate: YOLO tự động điều chỉnh

### 4. **Production Deployment**
- Sử dụng `best.pt` không phải `last.pt`
- Export sang ONNX nếu cần tối ưu inference
- Benchmark speed trước khi deploy

## 🔄 Export & Optimization

### Export sang ONNX (tối ưu inference)

```python
from yolo_model import YOLOFireDetector

yolo = YOLOFireDetector(model_path='path/to/best.pt')
yolo.model.export(format='onnx')
```

### Export sang TensorRT (GPU ultra-fast)

```python
yolo.model.export(format='engine')
```

## 📊 Performance Metrics

### Mong đợi kết quả

| Metric | ML Models | YOLO Expected |
|--------|-----------|---------------|
| Accuracy | 60-73% | 85-95% |
| F1-Score | 50-73% | 85-95% |
| Inference (CPU) | <10ms | 50-200ms |
| Inference (GPU) | N/A | 1-5ms |

## 🎯 Use Cases

### Khi nào dùng YOLO?

✅ **Dùng YOLO khi:**
- Cần accuracy cao nhất
- Có GPU available
- Dataset đủ lớn (>500 images/class)
- Real-time không quá critical

❌ **Dùng ML truyền thống khi:**
- Cần inference cực nhanh trên CPU
- Tài nguyên hạn chế (embedded devices)
- Cần model size nhỏ
- Cần interpretability

## 📚 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLO Classification](https://docs.ultralytics.com/tasks/classify/)
- [Model Export](https://docs.ultralytics.com/modes/export/)

## 🤝 Integration Examples

### Python API

```python
from yolo_model import YOLOFireDetector

# Load model
yolo = YOLOFireDetector(model_path='path/to/best.pt')

# Predict single image
result = yolo.predict('test_image.jpg')
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")

# Predict batch
results = [yolo.predict(img) for img in image_list]
```

### REST API (Web App)

```bash
curl -X POST http://localhost:8085/upload \
  -F "file=@test_image.jpg"
```

Response includes YOLO prediction alongside ML models!

---

**🔥 Chúc bạn training thành công!**

