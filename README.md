# 🌸 Phân Loại Ảnh Hoa AlexNet PyTorch (Rose, Daisy, Lily)

Dự án phân loại 3 loại hoa (**Hoa Hồng / Rose**, **Hoa Cúc / Daisy**, **Hoa Ly / Lily**) bằng mô hình
Deep Learning **AlexNet** (PyTorch) kết hợp ứng dụng web **FastAPI** + giao diện **Glassmorphism** hiện đại.

> **Kết quả hiện tại (model đang triển khai — AlexNet transfer learning):**
> **Accuracy 94.20%** · **Macro F1 94.52%** trên 707 ảnh test.
> Chi tiết: [`docs/training_report.md`](docs/training_report.md)

---

## 📌 Tính Năng Nổi Bật

- **Mô hình AlexNet (PyTorch)**: hỗ trợ 2 chế độ huấn luyện — *from scratch* và *transfer learning*
  từ ImageNet (`config.USE_PRETRAINED`), kèm **early stopping** và **ReduceLROnPlateau**.
- **Cấu trúc modular theo chuẩn ML Engineering**: `src/data/`, `src/models/`, `src/pipelines/`, `src/utils/`.
- **Tải dữ liệu tự động** từ Kaggle và lọc đúng 3 lớp hoa (`download_data.py`).
- **REST API (FastAPI)**: dự đoán từ ảnh upload, trả về xác suất của cả 3 lớp, có kiểm tra dữ liệu đầu vào.
- **Giao diện web hiện đại**: Dark mode + Glassmorphism, kéo & thả ảnh, **3 ảnh mẫu hoa thật** bấm 1 lần để thử,
  thanh phần trăm độ tin cậy theo thời gian thực.
- **Bộ test `pytest`**: **69 passed / 1 skipped** (~8 giây) — model, transforms, checkpoint, dataloader,
  predictor, downloader và API.
- **Notebook mỏng**: `notebooks/alexnet_flower_classification.ipynb` chỉ import từ `src/`, không lặp lại logic.

---

## 📊 Thông Tin Bộ Dữ Liệu

Dữ liệu lấy từ Kaggle rồi trích xuất 3 lớp:

- **Kaggle handle**: `utkarshsaxenadn/flower-classification-5-classes-roselilyetc` (Version 1)
- **Chia sẵn bởi dataset**: 3.000 train · 1.500 validation · 707 test — tổng **5.207 ảnh**

| Lớp | Train | Validation | Test |
|---|---|---|---|
| 🌼 `daisy` (Hoa Cúc) | 1.000 | 500 | 260 |
| 🌺 `lily` (Hoa Ly) | 1.000 | 500 | 350 |
| 🌹 `rose` (Hoa Hồng) | 1.000 | 500 | **97** |
| **Tổng** | **3.000** | **1.500** | **707** |

> ⚠️ `test/rose` chỉ có 97 ảnh (lệch khá nhiều so với 2 lớp còn lại) → khi so sánh các phiên bản model
> nên ưu tiên **macro F1** thay vì accuracy.

---

## 🏆 Kết Quả Thực Tế (tập test — 707 ảnh)

| Chỉ số | AlexNet from scratch | **AlexNet transfer learning** (đang dùng) |
|---|---|---|
| **Accuracy** | 0.8034 | **0.9420** |
| **Macro F1** | 0.8038 | **0.9452** |
| Macro Precision | 0.78 | **0.95** |
| Macro Recall | 0.84 | **0.94** |
| Val Accuracy tốt nhất | 0.8500 | **0.9660** (epoch 13/15) |

**Chi tiết theo lớp (transfer learning):**

| Lớp | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `daisy` (Hoa Cúc) | 0.97 | 0.90 | 0.93 | 260 |
| `lily` (Hoa Ly) | 0.92 | 0.97 | 0.94 | 350 |
| `rose` (Hoa Hồng) | 0.95 | 0.97 | 0.96 | 97 |

**Confusion matrix** (hàng = nhãn thật, cột = nhãn dự đoán):

| true ↓ / pred → | `daisy` | `lily` | `rose` | Tổng |
|---|---|---|---|---|
| `daisy` | **233** | 26 | 1 | 260 |
| `lily` | 7 | **339** | 4 | 350 |
| `rose` | 0 | 3 | **94** | 97 |

> Điểm yếu còn lại duy nhất: 26/260 ảnh hoa cúc bị đoán thành hoa ly (recall `daisy` = 0.90).
> Toàn bộ chi tiết huấn luyện, so sánh 3 phiên bản và phân tích nằm ở
> [`docs/training_report.md`](docs/training_report.md).

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
DL/
├── config.py                            # Cấu hình trung tâm (đường dẫn, hyperparameter, seed)
├── download_data.py                     # CLI tải & lọc dữ liệu Kaggle
├── requirements.txt                     # Thư viện chạy (đã pin phiên bản)
├── requirements-dev.txt                 # Thư viện phát triển (pytest, httpx)
├── pytest.ini                           # Cấu hình pytest
├── REFACTOR_PLAN.md                     # Kế hoạch & trạng thái từng hạng mục sửa chữa
├── README.md
├── docs/
│   ├── baseline.md                      # Số liệu baseline trước khi sửa + tiêu chí nghiệm thu
│   ├── training_report.md               # Báo cáo train lại & so sánh 3 phiên bản model
│   ├── CLEANUP_REPORT.md                # Báo cáo quét & dọn code/file thừa (chi tiết)
│   └── BAO_CAO.md                       # Báo cáo tổng hợp đồ án
├── notebooks/
│   └── alexnet_flower_classification.ipynb   # Notebook mỏng, import từ src/
├── scripts/
│   └── make_sample_images.py            # Tạo 3 ảnh mẫu thật cho web app
├── samples/                             # Ảnh mẫu hoa thật (rose/daisy/lily.jpg)
├── checkpoints/                         # Weights .pth (bị .gitignore, không commit)
├── outputs/                             # Log, history.json, biểu đồ (bị .gitignore)
├── flower_data_3classes/                # Dữ liệu (bị .gitignore, tải bằng download_data.py)
├── src/
│   ├── data/
│   │   ├── downloader.py                # Nguồn duy nhất của logic tải/lọc Kaggle
│   │   ├── dataset.py                   # Dataset & tạo ImageFolder
│   │   ├── dataloader.py                # Factory DataLoader + rào chắn thiếu split
│   │   └── transforms.py                # Augmentation & normalize (ImageNet stats)
│   ├── models/
│   │   ├── alexnet.py                   # AlexNetFromScratch + get_alexnet_pretrained
│   │   └── builder.py                   # build_model / load_checkpoint_data / load_trained_model
│   ├── pipelines/
│   │   ├── trainer.py                   # Training loop, early stopping, lưu artifact
│   │   ├── evaluator.py                 # Đánh giá + confusion matrix
│   │   └── predictor.py                 # FlowerPredictor (inference)
│   └── utils/
│       ├── console.py                   # Ép UTF-8 cho console Windows
│       ├── logger.py                    # setup_logging (cấu hình log)
│       ├── plotting.py                  # configure_matplotlib_backend()
│       └── seed.py                      # set_seed (tái lập kết quả)
├── tests/                               # 8 module test + helpers (chạy: pytest -q)
└── app/
    ├── main.py                          # FastAPI app (lifespan, validate upload, REST API)
    ├── static/                          # style.css, script.js (Glassmorphism UI)
    └── templates/index.html             # Jinja2 template
```

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

### 0. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

> Yêu cầu: **Python 3.10+** (đã kiểm thử trên Python 3.12). GPU tùy chọn — code tự dùng CUDA nếu có,
> ngược lại chạy CPU.

### 1. Cài đặt thư viện phụ thuộc

```bash
pip install -r requirements.txt        # chạy dự án
pip install -r requirements-dev.txt    # thêm pytest + httpx nếu muốn chạy test
```

### 2. Tải & lọc bộ dữ liệu hoa

```bash
python download_data.py                # tải nếu chưa có
python download_data.py --force        # xóa dữ liệu cũ rồi tải lại
```

Cần cấu hình Kaggle API (`~/.kaggle/kaggle.json`). Script sẽ tải dataset, trích đúng 3 lớp
`daisy`, `lily`, `rose` và tạo cấu trúc `flower_data_3classes/{train,val,test}/<lớp>/`.

### 3. Huấn luyện mô hình

```bash
python -m src.pipelines.trainer
```

- Mặc định `config.USE_PRETRAINED = True` (transfer learning) — đây là cấu hình cho kết quả 94.20% ở trên.
- Muốn train from scratch: sửa `USE_PRETRAINED = False` trong `config.py` (kỳ vọng ~80–85%).
- Model tốt nhất (theo `val_acc`) lưu vào `checkpoints/best_alexnet_flowers.pth` kèm metadata
  (`arch`, `use_pretrained`, `classes`, `val_acc`, `trained_at`, …).
- Đồ thị & lịch sử huấn luyện: `outputs/training_curves.png`, `outputs/history.json`.
- Early stopping: dừng nếu `val_acc` không cải thiện sau `config.EARLY_STOPPING_PATIENCE` epoch (mặc định 5).
- **Thời gian tham khảo**: ~9,5 phút cho 15 epoch trên Quadro T1000.

### 4. Đánh giá mô hình trên tập test

```bash
python -m src.pipelines.evaluator
```

In ra `classification_report` (precision/recall/F1 từng lớp) và lưu confusion matrix vào
`outputs/confusion_matrix.png`.

### 5. (Tùy chọn) Tạo ảnh mẫu cho web app

```bash
python scripts/make_sample_images.py
```

Tạo `samples/{rose,daisy,lily}.jpg` (~23KB tổng) từ tập test để bấm thử nhanh trên giao diện web.

### 6. Khởi chạy Web App & REST API

```bash
python -m uvicorn app.main:app --reload
```

Mở trình duyệt: **http://127.0.0.1:8000** · Tài liệu API tự động: **http://127.0.0.1:8000/docs**

### 7. Chạy test

```bash
pytest -q
```

---

## 🌐 REST API

| Method | Endpoint | Mô Tả |
|---|---|---|
| `GET` | `/` | Giao diện Web UI |
| `GET` | `/api/health` | Trạng thái server + metadata model (arch, val_acc, classes, trained_at…) |
| `GET` | `/api/samples` | Danh sách ảnh mẫu hoa thật dùng cho UI |
| `POST` | `/api/predict` | Upload ảnh hoa → JSON kết quả dự đoán + xác suất 3 lớp |
| `GET` | `/samples/{file}.jpg` | File ảnh mẫu tĩnh |
| `GET` | `/docs` | Swagger UI (FastAPI tự sinh) |

### Mã lỗi của `POST /api/predict`

| HTTP | Khi nào | Ví dụ thông báo |
|---|---|---|
| `400` | Sai content-type | `File 'n.txt' không phải ảnh (content-type: text/plain).` |
| `400` | Đuôi file không hỗ trợ | `Định dạng '.gif' không được hỗ trợ. Chỉ nhận: .jpg, .jpeg, .png, .bmp, .webp.` |
| `400` | Ảnh hỏng / không đọc được | `Không đọc được nội dung ảnh. File có thể bị hỏng hoặc không phải là ảnh.` |
| `400` | File rỗng | `File rỗng, không có dữ liệu ảnh.` |
| `413` | Ảnh > 10MB (`config.MAX_UPLOAD_SIZE_MB`) | `Ảnh vượt quá giới hạn 10MB (kích thước thực tế: 12.4MB).` |
| `503` | Server chưa load xong model | `Mô hình chưa được khởi tạo xong. Vui lòng thử lại sau vài giây.` |

Ảnh JPEG chụp từ điện thoại được tự động chỉnh hướng theo EXIF trước khi dự đoán.

### Ví dụ `POST /api/predict`

```bash
curl -X POST "http://127.0.0.1:8000/api/predict" -F "file=@samples/rose.jpg"
```

```json
{
  "success": true,
  "data": {
    "prediction": "rose",
    "prediction_vi": "Hoa Hồng (Rose)",
    "confidence": 0.9998,
    "confidence_percentage": 99.98,
    "model_trained": true,
    "filename": "rose.jpg",
    "image_size": [225, 225],
    "probabilities": {
      "daisy": { "name_en": "daisy", "name_vi": "Hoa Cúc (Daisy)", "probability": 0.0001, "percentage": 0.01 },
      "lily":  { "name_en": "lily",  "name_vi": "Hoa Ly (Lily)",    "probability": 0.0001, "percentage": 0.01 },
      "rose":  { "name_en": "rose",  "name_vi": "Hoa Hồng (Rose)",  "probability": 0.9998, "percentage": 99.98 }
    }
  }
}
```

> `model_trained: false` nghĩa là chưa tìm thấy checkpoint — kết quả không có ý nghĩa, hãy chạy bước 3.
> Giao diện web cũng hiển thị banner cảnh báo trong trường hợp này.

### Ví dụ `GET /api/health`

```json
{
  "status": "online",
  "model_loaded": true,
  "model_trained": true,
  "model_checkpoint_exists": true,
  "device": "cuda",
  "target_classes": ["daisy", "lily", "rose"],
  "loaded_classes": ["daisy", "lily", "rose"],
  "model": {
    "arch": "alexnet",
    "use_pretrained": true,
    "trained_at": "2026-10-18T22:12:08",
    "best_epoch": 13,
    "best_val_acc": 0.966,
    "num_classes": 3,
    "image_size": 224,
    "batch_size": 32,
    "learning_rate": 0.0001
  },
  "samples_available": ["daisy", "lily", "rose"],
  "max_upload_mb": 10,
  "allowed_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
}
```

---

## ⚙️ Cấu Hình (`config.py`)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `TARGET_CLASSES` | `['daisy', 'lily', 'rose']` | Thứ tự lớp |
| `IMAGE_SIZE` | 224 | Đầu vào AlexNet |
| `BATCH_SIZE` | 32 | |
| `NUM_EPOCHS` | 15 | |
| `LEARNING_RATE` | 1e-4 | Adam |
| `WEIGHT_DECAY` | 1e-4 | |
| `EARLY_STOPPING_PATIENCE` | 5 | 0/None = tắt |
| `USE_PRETRAINED` | `True` | `False` = train from scratch |
| `SEED` | 42 | Tái lập kết quả |
| `MAX_UPLOAD_SIZE_MB` | 10 | Giới hạn upload của API |

---

## 🧪 Kiểm Thử

```bash
pytest -q
# 69 passed, 1 skipped in ~8s
```

Bao gồm: kiến trúc & forward pass của model, transforms/normalize, format checkpoint (cũ & mới),
thứ tự lớp + `use_pretrained` đọc từ checkpoint, `FlowerPredictor`, bộ lọc dữ liệu Kaggle,
rào chắn thiếu split, và **toàn bộ REST API** (`tests/test_api.py` — validate upload, mã lỗi, health).

---

## 📚 Tài Liệu Liên Quan

| File | Nội dung |
|---|---|
| [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md) | Kế hoạch 6 giai đoạn sửa chữa + trạng thái từng hạng mục |
| [`docs/CLEANUP_REPORT.md`](docs/CLEANUP_REPORT.md) | Báo cáo quét & dọn code/file thừa (từng mục, kèm lý do giữ/xoá) |
| [`docs/baseline.md`](docs/baseline.md) | Baseline đo trước khi sửa và tiêu chí nghiệm thu |
| [`docs/training_report.md`](docs/training_report.md) | So sánh 3 phiên bản model + phân tích |
| [`docs/BAO_CAO.md`](docs/BAO_CAO.md) | Báo cáo tổng hợp đồ án |
| [`samples/README.md`](samples/README.md) | Nguồn gốc 3 ảnh mẫu và cách tạo lại |

---

## ⚠️ Hạn Chế & Hướng Phát Triển

**Hạn chế**
1. `test/rose` chỉ có 97 ảnh → đánh giá lớp này kém ổn định (sai số chuẩn ≈ 1,5 pp trên 707 ảnh).
2. 26/260 ảnh `daisy` bị nhận nhầm thành `lily` (recall `daisy` = 0.90).
3. AlexNet from scratch chỉ đạt ~80–85% — bộ dữ liệu 3.000 ảnh là quá nhỏ cho kiến trúc này.
4. Baseline cũ có lỗi: bịa ảnh ngẫu nhiên khi thiếu dữ liệu và tự đoán tên lớp theo thư mục
   (đã sửa triệt để ở Phase 1 — xem `REFACTOR_PLAN.md`).

**Hướng phát triển**
1. Unfreeze có chọn lọc / dùng LR nhỏ hơn cho phần `features` để tránh phá pre-trained weights.
2. Bổ sung dữ liệu `rose` để đánh giá công bằng hơn giữa các lớp.
3. Thử `WeightedRandomSampler` hoặc class weight cho `CrossEntropyLoss`.
4. Export ONNX để tăng tốc inference.
5. So sánh với các backbone hiện đại hơn (ResNet / EfficientNet) — *lưu ý: yêu cầu của đồ án là giữ AlexNet*.

---

## 🗂️ Ghi Chú Về Artifact

- `checkpoints/*.pth` (khoảng **245MB** cho bản mới — xem ghi chú bên dưới), `flower_data_3classes/`,
  `outputs/*` đều nằm trong `.gitignore` → **không** được commit lên Git.
- Sau khi `git clone`, cần chạy `python download_data.py` rồi `python -m src.pipelines.trainer`
  để tái tạo dữ liệu và model (kết quả có thể lệch ±1 pp do seed/GPU).
- Riêng 3 ảnh mẫu trong `samples/` (~23KB) **được commit** để giao diện web hoạt động ngay.
- **Kích thước checkpoint**: bản mới chỉ ~245MB vì trainer **không còn lưu `optimizer_state_dict`**
  (trạng thái Adam nặng gấp ~3 lần trọng số và không bao giờ được dùng lại — dự án không hỗ trợ
  resume training). Checkpoint cũ (652MB) vẫn load được bình thường.
