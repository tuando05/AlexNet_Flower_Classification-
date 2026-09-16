# 🌸 Phân Loại Ảnh Hoa AlexNet PyTorch (Rose, Daisy, Lily)

Dự án phân loại 3 loại hoa (**Hoa Hồng / Rose**, **Hoa Cúc / Daisy**, **Hoa Ly / Lily**) sử dụng mô hình Deep Learning **AlexNet** (PyTorch) kết hợp với ứng dụng Web giao diện **FastAPI** + **Glassmorphic UI** hiện đại.

---

## 📌 Tính Năng Nổi Bật

- **Mô hình Deep Learning AlexNet**: Hỗ trợ 2 tùy chọn huấn luyện (AlexNet From Scratch & Pre-trained Transfer Learning từ ImageNet).
- **Cấu trúc Mã nguồn Modular**: Chia nhỏ dự án theo chuẩn ML Engineering (`src/data/`, `src/models/`, `src/pipelines/`).
- **Kịch bản Tải Dữ liệu Tự động**: Tải và lọc bộ dữ liệu Kaggle 3 loại hoa tự động thông qua `download_data.py`.
- **REST API với FastAPI**: Endpoints chuẩn xác nhận ảnh upload và trả về kết quả dự đoán kèm tỉ lệ phần trăm xác suất.
- **Giao diện Web UI Hiện đại**:
  - Giao diện Dark Mode + hiệu ứng Glassmorphism sống động.
  - Kéo & thả tệp ảnh hoa trực quan.
  - Mẫu ảnh có sẵn thử nghiệm nhanh chỉ với 1 cú click.
  - Thanh phần trăm độ tin cậy (Confidence Progress Bars) thời gian thực.

---

## 📊 Thông Tin Bộ Dữ Liệu (Dataset)

Dự án sử dụng dữ liệu được tích hợp từ Kaggle:
- **Kaggle Handle**: `utkarshsaxenadn/flower-classification-5-classes-roselilyetc` (Bản **Version 1**)
- **Các lớp hoa được trích xuất (3 classes)**:
  - 🌼 **Daisy (Hoa Cúc)**: 1,000 ảnh Train | 500 ảnh Val | 260 ảnh Test
  - 🌺 **Lily (Hoa Ly)**: 1,000 ảnh Train | 500 ảnh Val | 350 ảnh Test
  - 🌹 **Rose (Hoa Hồng)**: 1,000 ảnh Train | 500 ảnh Val | 97 ảnh Test
- **Tổng số lượng**: **5,207** ảnh chất lượng cao.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
DL/
├── config.py                           # Cấu hình tham số trung tâm
├── download_data.py                    # Kịch bản tải & lọc dữ liệu Kaggle
├── requirements.txt                    # Thư viện phụ thuộc
├── README.md                           # Tài liệu hướng dẫn dự án
├── .gitignore                          # Cấu hình bỏ qua Git
├── .gitattributes                      # Cấu hình chuẩn hóa dòng Git
├── notebooks/                          # Jupyter / Colab Notebooks
│   ├── .gitkeep
│   └── alexnet_flower_classification.ipynb
├── outputs/                            # Thư mục chứa báo cáo & đồ thị đánh giá
│   ├── .gitkeep
│   └── confusion_matrix.png
├── checkpoints/                        # Thư mục lưu weights (.pth)
│   └── .gitkeep
├── src/                                # Core Modular ML Framework
│   ├── __init__.py                     # High-level exports
│   ├── data/                           # Module Quản lý Dữ liệu
│   │   ├── transforms.py               # Data Augmentation & Preprocessing
│   │   ├── dataset.py                  # Dataset Preparation & Kaggle Pipeline
│   │   └── dataloader.py               # PyTorch DataLoaders Factory
│   ├── models/                         # Module Kiến trúc Model
│   │   ├── alexnet.py                  # AlexNet Scratch & Pretrained Transfer
│   │   └── builder.py                  # Model Builder Factory
│   └── pipelines/                      # Module Quy trình Xử lý ML
│       ├── trainer.py                  # Training Loop Pipeline
│       ├── evaluator.py                # Evaluation & Confusion Matrix Pipeline
│       └── predictor.py                # Inference Engine
└── app/                                # FastAPI Web Application
    ├── main.py                         # REST API & Routes
    ├── static/                         # Giao diện CSS & JS (Glassmorphism)
    └── templates/                      # HTML5 Templates
```

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Cài đặt các thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

### 2. Tải & Lọc Bộ Dữ Liệu Hoa (Kaggle v1)

Chạy kịch bản tự động tải bộ dữ liệu từ Kaggle và trích xuất đúng 3 loại hoa (`daisy`, `lily`, `rose`):

```bash
python download_data.py
```
*(Nếu muốn xóa dữ liệu cũ và ép tải/giải nén lại từ đầu, thêm cờ `--force`):*
```bash
python download_data.py --force
```

### 3. Huấn luyện mô hình PyTorch

Chạy quy trình huấn luyện AlexNet (mô hình tốt nhất sẽ tự động được lưu vào `checkpoints/best_alexnet_flowers.pth`):

```bash
python -m src.pipelines.trainer
```

### 4. Đánh giá mô hình & Xuất biểu đồ

Đánh giá mô hình trên tập dữ liệu Test và xuất biểu đồ Confusion Matrix vào `outputs/confusion_matrix.png`:

```bash
python -m src.pipelines.evaluator
```

### 5. Khởi chạy Web Application & REST API

Khởi chạy Uvicorn Web Server:

```bash
python -m uvicorn app.main:app --reload
```

Sau khi máy chủ khởi chạy thành công, truy cập ứng dụng tại trình duyệt:
👉 **`http://127.0.0.1:8000`**

---

## 🌐 Chi Tiết REST API Endpoints

| Method | Endpoint | Mô Tả |
|---|---|---|
| `GET` | `/` | Trang chủ giao diện Web UI Dashboard |
| `GET` | `/api/health` | Kiểm tra trạng thái máy chủ, model & danh sách lớp |
| `POST` | `/api/predict` | Upload file ảnh hoa và nhận JSON kết quả dự đoán |

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
