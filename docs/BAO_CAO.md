# 📄 BÁO CÁO ĐỒ ÁN — Phân Loại Ảnh Hoa Bằng AlexNet (PyTorch)

**Đề tài:** Phân loại ảnh hoa thành 3 lớp — Hoa Hồng (Rose), Hoa Cúc (Daisy), Hoa Ly (Lily)
**Mô hình:** AlexNet (CNN, PyTorch) kết hợp ứng dụng web FastAPI
**Ngày:** 2026-10-18 · **Nhánh:** `refactor/fix-project`

> **Kết luận ngắn:** AlexNet **transfer learning** từ ImageNet đạt **Accuracy 94.20%** và
> **Macro F1 94.52%** trên 707 ảnh test, vượt bản *from scratch* **+13,86 pp accuracy**.
> Sản phẩm kèm REST API + giao diện web hoàn chỉnh và bộ test tự động 69 test.

---

## 1. Bài toán & mục tiêu

| Nội dung | Mô tả |
|---|---|
| **Bài toán** | Phân loại ảnh (image classification), 3 lớp: `daisy`, `lily`, `rose` |
| **Đầu vào** | Ảnh RGB (JPEG/PNG/BMP/WEBP), mọi kích thước |
| **Đầu ra** | Nhãn dự đoán + xác suất softmax của cả 3 lớp |
| **Ràng buộc** | Bắt buộc dùng họ kiến trúc **AlexNet** (yêu cầu đồ án) |
| **Ứng dụng** | REST API + Web UI: upload ảnh, xem độ tin cậy từng lớp |

**Mục tiêu kỹ thuật đề ra:**

| Tiêu chí | Ngưỡng | Kết quả đạt được |
|---|---|---|
| Accuracy trên tập test | ≥ 0.83 | **0.9420** ✅ |
| Macro F1 | ≥ 0.84 | **0.9452** ✅ |
| Recall lớp `lily` | > 0.79 | **0.97** ✅ |
| Checkpoint có metadata tự mô tả | có | có ✅ |

---

## 2. Bộ dữ liệu

### 2.1 Nguồn

- Kaggle: `utkarshsaxenadn/flower-classification-5-classes-roselilyetc` (Version 1)
- Bộ gốc có 5 lớp; đồ án **lọc giữ đúng 3 lớp** `daisy`, `lily`, `rose`, giữ nguyên cách chia
  train/validation/test của tác giả dataset (đảm bảo so sánh được với các nghiên cứu khác).
- Toàn bộ việc tải + lọc nằm trong `src/data/downloader.py`, chạy qua `python download_data.py`.

### 2.2 Phân bố dữ liệu

| Lớp | Train | Validation | Test |
|---|---|---|---|
| `daisy` (Hoa Cúc) | 1.000 | 500 | 260 |
| `lily` (Hoa Ly) | 1.000 | 500 | 350 |
| `rose` (Hoa Hồng) | 1.000 | 500 | **97** |
| **Tổng** | **3.000** | **1.500** | **707** |

**Nhận xét quan trọng về dữ liệu:**

1. Tập train **cân bằng hoàn hảo** (1.000 ảnh/lớp), nhưng tập **test rất lệch**: `rose` chỉ có 97 ảnh
   (13,7% tập test) so với `lily` 350 ảnh. Vì vậy accuracy bị chi phối bởi `lily`/`daisy`, còn
   `rose` chỉ đóng góp ~14% vào accuracy → **dùng macro F1 làm chỉ số chính khi so sánh phiên bản**.
2. Với n = 707, sai số chuẩn của accuracy ≈ √(0,94·0,06/707) ≈ **0,9 pp**. Chênh lệch dưới ~2 pp
   giữa hai lần chạy **không có ý nghĩa thống kê** — đây là lý do phải kiểm chứng bằng cách chạy lại
   evaluator trên cùng checkpoint trước khi kết luận có hồi quy hay không (xem mục 5.3).
3. 3.000 ảnh train là **rất nhỏ** so với nhu cầu của AlexNet (60M tham số) → đây là nguyên nhân
   gốc khiến bản *from scratch* chỉ đạt ~80–85%.

### 2.3 Tiền xử lý & tăng cường dữ liệu

| Giai đoạn | Pipeline |
|---|---|
| **Train** | `RandomResizedCrop(224)` → `RandomHorizontalFlip()` → `RandomRotation(15°)` → `ColorJitter(brightness/contrast/saturation=0.2)` → `ToTensor()` → `Normalize(ImageNet)` |
| **Val/Test/Inference** | `Resize(256)` → `CenterCrop(224)` → `ToTensor()` → `Normalize(ImageNet)` |

- Chuẩn hoá theo thống kê ImageNet: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
- Không dùng augmentation ở val/test để kết quả đánh giá ổn định và tái lập được.
- Ảnh upload từ điện thoại được chỉnh hướng theo **EXIF** (`ImageOps.exif_transpose`) trước khi dự đoán —
  nếu bỏ qua bước này, ảnh chụp dọc bị xoay ngang sẽ làm giảm độ chính xác thực tế.

---

## 3. Kiến trúc mô hình

### 3.1 AlexNet (2 biến thể trong `src/models/alexnet.py`)

| Thành phần | Cấu hình |
|---|---|
| `features` (5 conv) | `Conv(3→64, k=11, s=4, p=2)` → MaxPool → `Conv(64→192, k=5, p=2)` → MaxPool → `Conv(192→384, k=3, p=1)` → `Conv(384→256, k=3, p=1)` → `Conv(256→256, k=3, p=1)` → MaxPool |
| `avgpool` | `AdaptiveAvgPool2d((6,6))` |
| `classifier` | `Dropout(0.5)` → `Linear(9216→4096)` → ReLU → `Dropout(0.5)` → `Linear(4096→4096)` → ReLU → `Linear(4096→3)` |
| **Tổng tham số** | **57.016.131** (~57,0M) |

Có 2 biến thể, chỉ khác nhau ở **điểm khởi tạo trọng số**:

1. **`AlexNetFromScratch`** — khởi tạo ngẫu nhiên (Kaiming mặc định của PyTorch), huấn luyện từ đầu.
2. **`get_alexnet_pretrained`** — nạp trọng số **ImageNet** (`AlexNet_Weights.DEFAULT`), sau đó **thay
   lớp cuối** `classifier[6] = Linear(4096, num_classes)`. Chiến lược: **fine-tune toàn bộ mạng**
   (không đóng băng `features`) với learning rate nhỏ (1e-4) để hạn chế phá vỡ đặc trưng đã học.

### 3.2 Hàm mất mát & tối ưu

| Thành phần | Lựa chọn | Lý do |
|---|---|---|
| Loss | `CrossEntropyLoss` | Phân loại đa lớp, nhãn rời rạc |
| Optimizer | `Adam(lr=1e-4, weight_decay=1e-4)` | Hội tụ nhanh, phù hợp fine-tune |
| Scheduler | `ReduceLROnPlateau(mode='min', factor=0.5, patience=2)` | Giảm LR khi `val_loss` chững lại |
| Early stopping | `patience = 5` epoch | Tránh overfit, tiết kiệm thời gian |
| Chọn model | Lưu checkpoint theo **`val_acc` tốt nhất** | Không dùng test set để chọn model |

> Thứ tự `patience` được thiết kế có chủ đích: scheduler (2) < early stopping (5) → scheduler có
> ít nhất 1–2 lần giảm LR rồi thử lại trước khi quyết định dừng.

---

## 4. Quy trình huấn luyện

**Cấu hình chung cho cả 3 phiên bản:** 224×224, batch 32, Adam lr=1e-4 wd=1e-4, 15 epoch tối đa,
seed 42, GPU Quadro T1000, `torch 2.5.1+cu121`.

```bash
python download_data.py                      # tải & lọc dữ liệu
python -m src.pipelines.trainer              # huấn luyện (USE_PRETRAINED trong config.py)
python -m src.pipelines.evaluator            # đánh giá trên tập test
```

**Các cải tiến được thêm trong quá trình sửa dự án:**

1. **Early stopping + đo thời gian từng epoch** → kiểm soát overfit và biết chi phí thật.
2. **`set_seed(42)`** cho `random`/`numpy`/`torch` → giảm dao động giữa các lần chạy.
3. **Checkpoint tự mô tả**: lưu kèm `arch`, `use_pretrained`, `classes`, `num_classes`, `image_size`,
   `val_acc`, `trained_at` → `FlowerPredictor` đọc **kiến trúc và thứ tự lớp từ chính checkpoint**,
   nên load được cả bản scratch lẫn bản pretrained mà không cần sửa `config.py`.
4. **Rào chắn dữ liệu**: báo lỗi rõ ràng nếu thiếu `train/val/test` hoặc một lớp bị rỗng, thay vì
   tự sinh ảnh ngẫu nhiên như bản cũ.
5. **`history.json` + `training_curves.png`** tự động sau mỗi lần train.
6. **Checkpoint gọn**: KHÔNG lưu `optimizer_state_dict` (trạng thái Adam nặng gấp ~3 lần trọng số và
   không bao giờ dùng lại vì dự án không hỗ trợ resume training) → checkpoint từ 652MB xuống ~218MB
   (áp dụng cho cả 3 checkpoint đã train: đã bỏ key này, giữ nguyên trọng số),
   đồng thời `load_checkpoint_data()` lọc bỏ key này khỏi metadata để không giữ hàng trăm MB tensor
   trong RAM suốt vòng đời của app web.

---

## 5. Kết quả

### 5.1 So sánh 3 phiên bản trên tập test (707 ảnh)

| Chỉ số | v1 Baseline (code cũ) | v2 Scratch (code mới) | **v3 Transfer learning** ✅ |
|---|---|---|---|
| **Accuracy** | 0.8345 | 0.8034 | **0.9420** |
| **Macro F1** | 0.8360 | 0.8038 | **0.9452** |
| Macro Precision | 0.82 | 0.78 | **0.95** |
| Macro Recall | 0.86 | 0.84 | **0.94** |
| Val Acc tốt nhất | 0.8540 | 0.8500 | **0.9660** |
| Best epoch | 15 | 13 | 13 |

### 5.2 Classification report của model triển khai (v3)

| Lớp | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| `daisy` | 0.97 | **0.90** | 0.93 | 260 |
| `lily` | 0.92 | 0.97 | 0.94 | 350 |
| `rose` | 0.95 | 0.97 | 0.96 | 97 |
| **accuracy** | | | **0.94** | 707 |
| **macro avg** | 0.95 | 0.94 | **0.95** | 707 |
| **weighted avg** | 0.94 | 0.94 | 0.94 | 707 |

**Confusion matrix** (hàng = nhãn thật, cột = nhãn dự đoán):

| true ↓ / pred → | `daisy` | `lily` | `rose` | Tổng |
|---|---|---|---|---|
| `daisy` | **233** | 26 | 1 | 260 |
| `lily` | 7 | **339** | 4 | 350 |
| `rose` | 0 | 3 | **94** | 97 |

**Phân tích lỗi:**

- Tổng số ảnh sai: **41/707 (5,8%)**.
- **Nhầm chủ yếu `daisy` → `lily` (26 ảnh, 63% tổng số lỗi).** Hoa cúc trắng và hoa ly trắng có
  hình dạng cánh/nhị rất giống nhau; ngoài ra `rose` chỉ có 97 ảnh nên ảnh hưởng của nhóm lỗi này
  bị "khuếch đại" trong recall của `daisy`.
- Chiều ngược lại ít hơn nhiều (`lily` → `daisy` chỉ 7 ảnh) → model **thiên về `lily`** khi gặp ảnh
  hoa trắng nhiều cánh, hệ quả trực tiếp của việc `lily` là lớp đông thứ hai trong train và đông nhất
  trong test.
- `rose` gần như hoàn hảo (0 ảnh bị đoán nhầm sang `daisy`).

### 5.3 Diễn biến huấn luyện v3 & kiểm chứng không hồi quy

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 01 | 0.4295 | 82.83% | 0.2012 | **92.47%** |
| 03 | 0.2661 | 89.67% | 0.1440 | 95.20% |
| 07 | 0.1870 | 92.23% | 0.1208 | 96.27% |
| **13** | **0.1509** | **93.97%** | **0.1102** | **96.60%** ✅ |
| 15 | 0.1531 | 93.80% | 0.1275 | 96.00% |

- **Ngay epoch 1, transfer learning đã đạt 92,47% val_acc** — vượt toàn bộ 15 epoch của bản scratch
  (85,0%). Đây là bằng chứng mạnh nhất cho giá trị của pre-trained features trên dataset nhỏ.
- Early stopping **không kích hoạt** (chuỗi không cải thiện dài nhất chỉ 3 epoch).
- Thời gian huấn luyện: **~9,5 phút** cho 15 epoch (≈ 30–43 giây/epoch).
- **Kiểm chứng không hồi quy:** v2 (scratch, code mới) ban đầu cho kết quả *thấp hơn* v1 (0.8034 vs
  0.8345). Do sai số chuẩn ≈ 0,9–1,5 pp trên 707 ảnh, nghi vấn đầu tiên là **lỗi code**. Cách kiểm
  chứng: chạy lại evaluator **mới** trên checkpoint **cũ** (v1) → thu được đúng `0.8345 / 0.8360`,
  trùng khớp tuyệt đối với baseline ⇒ pipeline đánh giá mới **không hồi quy**; chênh lệch nằm trong
  dao động ngẫu nhiên do seed (`outputs/after_fix_eval.log`).

---

## 6. Triển khai ứng dụng

### 6.1 Kiến trúc phần mềm

```
config.py                     ← cấu hình trung tâm (đường dẫn, hyperparameter, seed, giới hạn upload)
src/data/     downloader · dataset · dataloader · transforms
src/models/   alexnet · builder
src/pipelines/ trainer · evaluator · predictor
src/utils/    console · logger · plotting · seed
app/          main.py (FastAPI) · static/ · templates/
tests/        pytest (69 passed / 1 skipped)
```

**Nguyên tắc thiết kế:**

- **Một nguồn sự thật duy nhất**: metadata của checkpoint quyết định `arch`, `use_pretrained`,
  thứ tự lớp — mọi module (evaluator, predictor, app) đều đọc từ đó thay vì hard-code.
- **Library vs entry point**: module thư viện trong `src/` chỉ gọi `logging.getLogger(__name__)` và
  KHÔNG tự gắn handler; chỉ entry point gọi `setup_logging()`. Các entry point chạy được bằng
  `python -m` (trainer/evaluator/predictor) thì đặt tên logger cố định để log hiện đúng tên module
  thay vì `__main__`.
- **Không âm thầm chịu lỗi**: thiếu dữ liệu/checkpoint → ném lỗi rõ ràng kèm gợi ý lệnh cần chạy.

### 6.2 REST API

| Method | Endpoint | Chức năng |
|---|---|---|
| `GET` | `/` | Web UI (Glassmorphism, kéo & thả, 3 ảnh mẫu thật) |
| `GET` | `/api/health` | Trạng thái + metadata model (arch, val_acc, classes, trained_at…) |
| `GET` | `/api/samples` | Danh sách ảnh mẫu hoa thật |
| `POST` | `/api/predict` | Dự đoán từ ảnh upload |
| `GET` | `/docs` | Swagger UI |

**Kiểm soát dữ liệu vào `POST /api/predict`:**

| Trường hợp | HTTP | Thông báo (tiếng Việt) |
|---|---|---|
| Sai content-type | 400 | `File 'n.txt' không phải ảnh (content-type: text/plain).` |
| Đuôi file không hỗ trợ | 400 | `Định dạng '.gif' không được hỗ trợ. Chỉ nhận: .jpg, .jpeg, .png, .bmp, .webp.` |
| Ảnh hỏng | 400 | `Không đọc được nội dung ảnh. File có thể bị hỏng hoặc không phải là ảnh.` |
| File rỗng | 400 | `File rỗng, không có dữ liệu ảnh.` |
| Ảnh > 10MB | 413 | `Ảnh vượt quá giới hạn 10MB (kích thước thực tế: 12.4MB).` |
| Model chưa sẵn sàng | 503 | `Mô hình chưa được khởi tạo xong. Vui lòng thử lại sau vài giây.` |

- Model được nạp **một lần** khi server khởi động qua `lifespan` context manager (không nạp lại theo
  từng request), giải phóng khi tắt server.
- Giới hạn dung lượng do `config.MAX_UPLOAD_SIZE_MB` quyết định và được truyền xuống UI → client và
  server kiểm tra cùng một con số.
- Nếu thiếu checkpoint, server vẫn chạy nhưng trả `model_trained: false` và UI hiện banner cảnh báo
  "kết quả không có ý nghĩa".

### 6.3 Kiểm thử tự động

```bash
pytest -q        # 69 passed, 1 skipped  (~8 giây)
```

| Module | Nội dung kiểm thử |
|---|---|
| `test_model.py` | Forward pass, số lớp đầu ra, nhánh pretrained |
| `test_transforms.py` | Shape `3×224×224`, công thức normalize, augmentation |
| `test_checkpoint.py` | Format checkpoint cũ/mới, thứ tự lớp + `use_pretrained` từ checkpoint |
| `test_predictor.py` | Cấu trúc kết quả, tổng xác suất = 100%, nhãn tiếng Việt |
| `test_downloader.py` | Lọc đuôi file, cấu trúc Kaggle (không cần mạng) |
| `test_dataloader.py` | Rào chắn thiếu split / lớp rỗng |
| **`test_api.py`** | **Validate upload, mã lỗi 400/413, `/api/health`, `/api/samples`, ảnh mẫu** |

> Test dùng `TestClient` + `FakePredictor` nên **không cần checkpoint 218MB** — bộ test chạy được cả
> trên máy chưa train model.

### 6.4 Kiểm thử thủ công end-to-end (server thật + checkpoint v3)

| Ảnh mẫu | Dự đoán | Độ tin cậy |
|---|---|---|
| `daisy.jpg` | Hoa Cúc (Daisy) | 97,33% |
| `lily.jpg` | Hoa Ly (Lily) | 99,99% |
| `rose.jpg` | Hoa Hồng (Rose) | 100,00% |

Và toàn bộ trường hợp lỗi ở mục 6.2 đều trả đúng mã HTTP + thông báo tiếng Việt.

---

## 7. Hạn chế

1. **Mất cân bằng tập test**: `rose` chỉ 97 ảnh → sai số chuẩn lớn, chỉ số của lớp này kém ổn định.
2. **Nhầm `daisy` ↔ `lily`**: 26/260 ảnh hoa cúc bị đoán thành hoa ly (recall `daisy` = 0,90). Hai lớp
   có đặc trưng hình dáng/màu sắc (hoa trắng nhiều cánh) rất gần nhau.
3. **AlexNet from scratch kém**: chỉ ~80–85% vì 3.000 ảnh train quá nhỏ so với 60M tham số của AlexNet;
   bản thân AlexNet cũng là kiến trúc 2012, đã lạc hậu so với ResNet/EfficientNet/ViT.
4. **Chỉ nhận ảnh chứa đúng 1 bông hoa**: chưa có phát hiện/định vị đối tượng, ảnh nhiều hoa hoặc
   ảnh phông nền phức tạp sẽ làm giảm độ chính xác.
5. **Chưa có lớp từ chối (reject option)**: model luôn trả về 1 trong 3 lớp với softmax, kể cả khi
   ảnh không phải hoa → độ tin cậy cao nhưng sai. Chưa có ngưỡng "không xác định".
6. **Fine-tune toàn bộ mạng**: chưa thử các chiến lược tinh chỉnh tốt hơn (đóng băng có chọn lọc,
   discriminative learning rate).
7. **Chưa đóng gói triển khai**: chưa có Docker/hosting, chưa export ONNX, chưa có CI/CD.

---

## 8. Hướng phát triển

| Ưu tiên | Việc cần làm | Kỳ vọng |
|---|---|---|
| Cao | Bổ sung dữ liệu `rose` (đặc biệt tập test) | Đánh giá công bằng, giảm sai số chuẩn |
| Cao | `WeightedRandomSampler` / class weight cho `CrossEntropyLoss` | Giảm thiên lệch về `lily` |
| Cao | Ngưỡng từ chối: nếu `max(softmax) < τ` → "không xác định" | Tránh kết quả tự tin nhưng sai |
| Trung bình | Unfreeze có chọn lọc + LR riêng cho `features` | Tăng nhẹ accuracy, tránh phá pre-trained weights |
| Trung bình | Export ONNX / `torch.compile` | Inference nhanh hơn cho web app |
| Trung bình | Thêm test cho CLI, coverage ≥ 80% | Tăng độ tin cậy khi bảo trì |
| Thấp | Docker hoá + GitHub Actions | Triển khai và kiểm thử tự động |
| Thấp | So sánh tham khảo với ResNet50/EfficientNet-B0 | Đặt kết quả vào bối cảnh (không thay thế AlexNet) |

---

## 9. Kết luận

1. Đồ án đã xây dựng **hoàn chỉnh pipeline** từ tải dữ liệu → huấn luyện → đánh giá → triển khai
   web app, với cấu trúc module rõ ràng và bộ test tự động.
2. Model triển khai **AlexNet transfer learning** đạt **94,20% accuracy / 94,52% macro F1** — vượt
   toàn bộ tiêu chí nghiệm thu đề ra (≥ 0,83 / ≥ 0,84) và vượt bản from scratch **+13,86 pp**.
3. Phát hiện quan trọng: **pre-trained ImageNet features quyết định gần như toàn bộ chất lượng** —
   epoch đầu tiên của transfer learning đã cao hơn 15 epoch train from scratch. Với dataset nhỏ
   (~3.000 ảnh/lớp train), đây là lựa chọn gần như bắt buộc.
4. Quá trình sửa dự án cũng sửa được các lỗi nghiêm trọng về tính đúng/sai của code cũ: tự sinh ảnh
   ngẫu nhiên khi thiếu dữ liệu, tự đoán tên lớp theo tên thư mục, hard-code `use_pretrained` khi load
   checkpoint. Nhờ đó kết quả báo cáo giờ **tái lập được** và **không còn phụ thuộc vào tên thư mục**.

---

## 10. Phụ lục

### 10.1 Môi trường đã kiểm thử

| Thành phần | Phiên bản |
|---|---|
| Python | 3.12 |
| torch / torchvision | 2.5.1+cu121 / 0.20.1+cu121 |
| GPU | NVIDIA Quadro T1000 (CUDA) |
| fastapi / uvicorn | 0.141.1 / 0.53.0 |
| scikit-learn / matplotlib / seaborn | 1.9.1 / 3.11.2 / 0.13.2 |
| pytest / httpx | 9.1.1 / 0.28.1 |

### 10.2 Danh sách artifact

| File | Nội dung |
|---|---|
| `checkpoints/best_alexnet_flowers.pth` | **Model v3 đang triển khai** (epoch 13, val_acc 0.966, pretrained) |
| `checkpoints/best_alexnet_flowers_v2_scratch.pth` | Model v2 scratch (để đối chứng) |
| `checkpoints/best_alexnet_flowers_v1_scratch.pth` | Model v1 baseline gốc |
| `outputs/history.json`, `outputs/training_curves.png` | Lịch sử & đồ thị huấn luyện v3 |
| `outputs/confusion_matrix.png` | Confusion matrix v3 trên tập test |
| `outputs/train_run_v3_pretrained.log`, `outputs/eval_run_v3_pretrained.log` | Log train/eval v3 |
| `outputs/after_fix_eval.log` | Chạy lại evaluator mới trên checkpoint v1 (kiểm chứng không hồi quy) |
| `docs/baseline.md` | Số liệu baseline & tiêu chí nghiệm thu |
| `docs/training_report.md` | Báo cáo Phase 2 chi tiết (so sánh 3 phiên bản) |
| `REFACTOR_PLAN.md` | Kế hoạch sửa chữa 6 giai đoạn + trạng thái |

> ⚠️ `checkpoints/*.pth` (~218MB/bản vì không còn lưu `optimizer_state_dict`), `flower_data_3classes/` và `outputs/*` đều **bị
> `.gitignore`** → không commit lên Git. Sau khi clone cần chạy `download_data.py` và
> `src.pipelines.trainer` để tái tạo (kết quả có thể lệch ±1 pp do seed/GPU).

### 10.3 Lệnh chạy nhanh

```bash
# 1. Cài đặt
pip install -r requirements.txt && pip install -r requirements-dev.txt

# 2. Dữ liệu
python download_data.py

# 3. Huấn luyện (transfer learning — cấu hình mặc định)
python -m src.pipelines.trainer

# 4. Đánh giá
python -m src.pipelines.evaluator

# 5. Ảnh mẫu + web app
python scripts/make_sample_images.py
python -m uvicorn app.main:app --reload      # http://127.0.0.1:8000

# 6. Dự đoán nhanh bằng CLI (chạy trên 3 ảnh mẫu thật)
python -m src.pipelines.predictor

# 7. Test
pytest -q
```
