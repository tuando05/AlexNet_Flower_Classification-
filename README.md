# 🌸 Project Phân Loại Ảnh Hoa AlexNet PyTorch (Rose, Daisy, Lily)

Dự án phân loại 3 loại hoa (**Hoa Hồng**, **Hoa Cúc**, **Hoa Ly**) sử dụng mô hình Deep Learning **AlexNet** (PyTorch) kết hợp ứng dụng Web giao diện hiện đại (**FastAPI** + **Glassmorphic UI**).

---

## 📌 Tính Năng Nổi Bật

- **Mô hình AlexNet PyTorch**: Hỗ trợ 2 tùy chọn huấn luyện (AlexNet From Scratch & Pre-trained Transfer Learning từ ImageNet).
- **Quy trình ML tiêu chuẩn**: Tách biệt rõ ràng các mô-đun Data Processing, Training, Evaluation và Inference.
- **REST API với FastAPI**: Endpoint `/api/predict` nhận ảnh upload và trả về kết quả phân loại xác suất định dạng JSON.
- **Giao diện Web UI cao cấp**:
  - Giao diện Dark Mode + hiệu ứng Glassmorphism sống động.
  - Kéo & thả tệp ảnh hoa trực quan.
  - Mẫu ảnh có sẵn thử nghiệm nhanh chỉ với 1 cú click.
  - Thanh phần trăm độ tin cậy (Confidence Progress Bars) thời gian thực.

---

## 📁 Cấu Trúc Dự Án

```
DL/
├── alexnet_flower_classification.ipynb # Notebook gốc huấn luyện Colab
├── config.py                           # Cấu hình tham số dự án & Hyperparameters
├── requirements.txt                    # Khai báo các thư viện phụ thuộc
├── README.md                           # Hướng dẫn chi tiết
├── checkpoints/                        # Thư mục chứa weights model (.pth)
│   └── best_alexnet_flowers.pth
├── src/                                # Core Machine Learning Modules
│   ├── __init__.py
│   ├── dataset.py                      # Data Loading, Augmentation & Splitting
│   ├── model.py                        # Kiến trúc PyTorch AlexNet
│   ├── train.py                        # Script Huấn luyện mô hình
│   ├── evaluate.py                     # Script Đánh giá mô hình & vẽ Confusion Matrix
│   └── inference.py                    # Class dự đoán (FlowerPredictor)
└── app/                                # FastAPI Web Application
    ├── main.py                         # Web Server & REST API Routes
    ├── static/                         # Giao diện CSS & JavaScript
    │   ├── style.css
    │   └── script.js
    └── templates/                      # Giao diện HTML5
        └── index.html
```

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

### 2. Khởi tạo dữ liệu & Huấn luyện mô hình

Chạy kịch bản huấn luyện PyTorch (Tự động nạp dữ liệu hoặc tạo dữ liệu mẫu nếu chưa có):

```bash
python src/train.py
```

Sau khi hoàn tất, weights tốt nhất sẽ được tự động lưu vào `checkpoints/best_alexnet_flowers.pth`.

### 3. Đánh giá mô hình

Đánh giá mô hình trên tập dữ liệu Test và xuất biểu đồ Confusion Matrix:

```bash
python src/evaluate.py
```

### 4. Khởi chạy Ứng Dụng Web & REST API

Khởi chạy Uvicorn Web Server:

```bash
python -m uvicorn app.main:app --reload
```

Sau khi khởi chạy thành công, truy cập trình duyệt tại địa chỉ:
👉 **`http://127.0.0.1:8000`**

---

## 🌐 Chi Tiết REST API Endpoints

| Method | Endpoint | Mô Tả |
|---|---|---|
| `GET` | `/` | Trả về trang chủ giao diện Web UI |
| `GET` | `/api/health` | Kiểm tra trạng thái máy chủ & mô hình |
| `POST` | `/api/predict` | Tải ảnh hoa và nhận JSON kết quả dự đoán |

### Ví dụ JSON Response từ `/api/predict`:

```json
{
  "success": true,
  "data": {
    "prediction": "rose",
    "prediction_vi": "Hoa Hồng (Rose)",
    "confidence": 0.9854,
    "confidence_percentage": 98.54,
    "filename": "my_rose.jpg",
    "probabilities": {
      "daisy": {
        "name_en": "daisy",
        "name_vi": "Hoa Cúc (Daisy)",
        "percentage": 1.2
      },
      "lily": {
        "name_en": "lily",
        "name_vi": "Hoa Ly (Lily)",
        "percentage": 0.26
      },
      "rose": {
        "name_en": "rose",
        "name_vi": "Hoa Hồng (Rose)",
        "percentage": 98.54
      }
    }
  }
}
```
