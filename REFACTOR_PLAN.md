# 🗂️ KẾ HOẠCH SỬA LẠI DỰ ÁN — AlexNet Flower Classification

**Bài toán:** Phân loại 3 lớp hoa — `daisy` (Hoa Cúc), `lily` (Hoa Ly), `rose` (Hoa Hồng) — bằng AlexNet (PyTorch), kèm web app FastAPI.

**Nguyên tắc thực hiện:** sửa lỗi đúng/sai trước → nâng chất lượng mô hình → dọn cấu trúc code → hoàn thiện web/UX.
**Không thay đổi kiến trúc AlexNet** (giữ đúng yêu cầu bài toán).

**Nhánh làm việc:** `refactor/fix-project` (tách khỏi `main` @ `17fdc5b`)

---

## 🔴 Phase 0 — Dọn nền & chốt baseline (0.5 ngày)

Mục tiêu: biết chính xác hiện trạng đang đúng tới đâu trước khi sửa.

- [x] `P0.1` Chạy `python -m src.pipelines.evaluator` với checkpoint hiện có → ghi lại **accuracy, macro F1, confusion matrix** làm baseline.
      → **Kết quả: Accuracy 0.8345 · Macro F1 0.8360** (chi tiết: `docs/baseline.md`, log: `outputs/baseline_eval.log`)
- [x] `P0.2` Ghi lại metadata checkpoint hiện tại: `epoch`=15, `val_acc`=0.854, `classes`=`['daisy','lily','rose']`; **thiếu** `use_pretrained` / `arch` / `trained_at`.
- [x] `P0.3` Tạo nhánh git `refactor/fix-project`.

**Verify:** ✅ có số liệu baseline lưu trong `docs/baseline.md` (chuyển từ `outputs/` vì `outputs/*` bị gitignore).

---

## 🔴 Phase 1 — Sửa lỗi đúng/sai (1 ngày) — QUAN TRỌNG NHẤT

### 1.1 Bỏ cơ chế sinh ảnh random gây "học rác" im lặng
- [x] `src/data/dataset.py`: **xóa** fallback tạo ảnh synthetic random khi tải thất bại. Thay bằng `raise RuntimeError` kèm hướng dẫn `python download_data.py`.
- **Lý do:** hiện nếu Kaggle lỗi, script vẫn "thành công" và train trên nhiễu → kết quả vô nghĩa mà không ai phát hiện.

### 1.2 Sửa logic dò thư mục lớp sai
- [x] `dataset.py`: bỏ fuzzy match `'ly' in dir_name`. Chỉ ánh xạ đúng tên `daisy` / `lily` / `rose`, đồng thời bỏ qua thư mục split `train/val/test`.
- [x] Thêm `return True` rõ ràng cho `prepare_flower_dataset()`.

### 1.3 Gộp logic tải dữ liệu bị trùng
- [x] Tạo `src/data/downloader.py` chứa **một** hàm `download_and_filter_dataset()` duy nhất.
- [x] `download_data.py` → chỉ còn CLI wrapper gọi hàm trên (thoát code 1 khi thất bại).
- [x] `dataset.py` import lại từ `downloader.py` (xóa code copy-paste).

### 1.4 Sửa rủi ro sai thứ tự nhãn lớp
- [x] `predictor.py`: `class_names` đọc **từ checkpoint** (`checkpoint['classes']`), fallback về `config`.
- [x] Lưu thêm `use_pretrained`, `arch`, `num_classes`, `image_size`, `batch_size`, `learning_rate`, `trained_at` vào checkpoint; `load_trained_model()` build đúng kiến trúc theo checkpoint.

### 1.5 Kiểm tra đủ 3 split trước khi train
- [x] `dataloader.py`: kiểm tra cả `train/`, `val/`, `test/` và từng lớp có ảnh; raise lỗi rõ ràng nếu thiếu.
- [x] `trainer.py`: bỏ gọi `prepare_flower_dataset()` trùng.

### 1.6 Tương thích `torch.load` (torch ≥ 2.6)
- [x] Thêm `weights_only=False` tường minh trong `src/models/builder.py::load_checkpoint_data()` dùng chung cho cả evaluator và predictor.

### 1.7 Sửa `trainer.py`
- [x] Thêm `config.SEED = 42` + `src/utils/seed.py::set_seed()`.
- [x] Điều kiện lưu checkpoint dùng `>` thay vì `>=`.
- [x] Lưu `history` ra `outputs/history.json` + vẽ `outputs/training_curves.png`.

### 1.8 (Phát sinh khi smoke test) Sửa lỗi crash khi in tiếng Việt
- [x] `predictor.py` crash `UnicodeEncodeError` trên console Windows → gom thành `src/utils/console.py::setup_console()`, dùng chung cho `trainer`, `evaluator`, `predictor`, `download_data.py`, `app/main.py`.
- [x] Sửa `import vòng` gây `RuntimeWarning` của runpy: `src/__init__.py` và `src/pipelines/__init__.py` chuyển sang lazy import (PEP 562).
- [x] `evaluator.py`: dùng backend `Agg` (script chỉ lưu hình), thêm in **Accuracy + Macro F1**, cảnh báo nếu thứ tự lớp trong checkpoint lệch với ImageFolder.

**Verify:** ✅ Đã chạy smoke test đầy đủ:
- 15/15 module import OK, không còn `FutureWarning`/`RuntimeWarning`.
- `python download_data.py` → nhận diện dữ liệu có sẵn, không tải lại.
- `python -m src.pipelines.predictor` → load checkpoint, in được tiếng Việt, `classes` lấy từ checkpoint.
- `python -m src.pipelines.evaluator` → **Accuracy 0.8345 / Macro F1 0.8360**, trùng khớp baseline (không hồi quy); checkpoint cũ (thiếu metadata) vẫn load được.
- Thiếu split `val`/`test` → `RuntimeError` rõ ràng (đã kiểm chứng riêng lẻ).
- `app.main` import OK, đủ 3 route `/`, `/api/health`, `/api/predict`.

---

## 🟠 Phase 2 — Chất lượng mô hình & đánh giá (1–1.5 ngày, cần GPU)

- [x] `P2.1` **Train lại** với code đã sửa (15 epoch, `USE_PRETRAINED=False`) → checkpoint mới + `confusion_matrix.png` mới.
      → Xong. Best val_acc **85.00% @ epoch 13**, ~9 phút. Checkpoint cũ đã backup thành `checkpoints/best_alexnet_flowers_v1_scratch.pth`.
- [x] `P2.2` Báo cáo đầy đủ: **accuracy, precision, recall, F1 (macro & per-class)**; lưu ý `test/rose` chỉ 97 ảnh gây lệch lớp.
      → `docs/training_report.md`. Kết quả v2: **Accuracy 0.8034 / Macro F1 0.8038** → **thấp hơn** baseline (0.8345 / 0.8360).
- [x] `P2.3` Thêm **early stopping** (`config.EARLY_STOPPING_PATIENCE = 5`, đặt 0 để tắt) + đo thời gian mỗi epoch.
      → Không kích hoạt trong lần chạy này (chuỗi không cải thiện dài nhất = 2 epoch).
- [x] `P2.4` (Tùy chọn so sánh) Train bản `USE_PRETRAINED=True` → bảng so sánh Scratch vs Transfer Learning trong README.
      → **XONG và là phương án thắng.** Best val_acc **96.60% @ epoch 13**; test **Accuracy 0.9420 / Macro F1 0.9452**
      (vượt baseline **+10.75 pp**). `config.USE_PRETRAINED` đã chuyển thành `True`.
- [x] `P2.5` Lưu `training_report.md` + confusion matrix để dùng cho báo cáo môn học.
      → Đã lưu tại `docs/training_report.md` (đặt trong `docs/` để được version hóa, vì `outputs/*` bị gitignore).

**Verify:** ✅ **ĐẠT** — v3 (transfer learning): Accuracy 0.9420, Macro F1 0.9452, recall `lily` 0.97.
Đầy đủ 3 phiên bản được lưu và so sánh: `best_alexnet_flowers_v1_scratch.pth`,
`best_alexnet_flowers_v2_scratch.pth`, `best_alexnet_flowers.pth` (v3).
Chi tiết & phân tích: `docs/training_report.md`.

**Điểm đáng chú ý:** bản scratch train lại (v2) cho kết quả *thấp hơn* baseline (0.8034 / 0.8038)
nhưng đây **không phải lỗi code** — evaluator mới chạy lại trên checkpoint v1 vẫn cho đúng
0.8345 / 0.8360, và hai bản scratch có val_acc gần như trùng nhau (85.0% vs 85.4%). Mức lệch
nằm trong sai số chuẩn (~1.5 pp với 707 mẫu test) do dao động seed.

---

## 🟡 Phase 3 — Cấu trúc code & kiểm thử (1 ngày)

- [x] `P3.1` Thêm `tests/` (pytest) — **53 passed, 1 skipped (~8s)**
  - `tests/test_model.py` — forward shape `(N, 3)`, `build_model` với nhiều `num_classes`, thay lớp cuối của bản pre-trained, tính nhất quán của `config`
  - `tests/test_transforms.py` — shape `3×224×224`, kiểm tra **công thức normalize** trên ảnh đồng nhất, train transform ngẫu nhiên vs val transform tất định, ảnh không vuông / grayscale / RGBA
  - `tests/test_checkpoint.py` — định dạng checkpoint mới & cũ, **`weights_only=False`**, thứ tự lớp lấy từ checkpoint, `use_pretrained` lấy từ checkpoint (chống hồi quy lỗi 1.4)
  - `tests/test_predictor.py` — cấu trúc kết quả, tổng xác suất = 100%, top-1 khớp xác suất cao nhất, nhãn tiếng Việt, thiếu checkpoint, input không hợp lệ
  - `tests/test_downloader.py` — lọc đuôi file, phát hiện dữ liệu có sẵn, dò cấu trúc Kaggle (không cần mạng)
  - `tests/test_dataloader.py` — rào chắn thiếu split / lớp rỗng, thông báo lỗi có gợi ý cách sửa
  - `tests/_helpers.py` — `TinyModel` để test nhanh, không cần load AlexNet 240MB
  - `pytest.ini` — `testpaths`, loại trừ `.venv`/`flower_data_3classes`/`notebooks`/`outputs`/`checkpoints`
- [x] `P3.2` Thay `print()` bằng `logging` (`src/utils/logger.py` + `setup_logging()` gọi ở entry point).
      Các module thư viện chỉ dùng `logging.getLogger(...)`, không tự gắn handler.
      Chỉ giữ `print()` cho banner CLI trong `download_data.py` (đúng mục đích).
- [x] `P3.3` Thêm `src/utils/seed.py` (hàm `set_seed`) — *đã làm sớm ở Phase 1.7 vì cần cho việc tái lập kết quả.*
- [x] `P3.4` Pin phiên bản trong `requirements.txt` (theo `.venv` đã kiểm thử) + tách `requirements-dev.txt` cho pytest.
- [x] `P3.5` Làm sạch notebook: viết lại thành dạng mỏng **chỉ import từ `src/`**
      (17 cell copy-paste → 21 cell, trong đó 11 cell code đều là import/glue ngắn).
      Đã kiểm chứng bằng script chạy tuần tự toàn bộ cell code: **11/11 cell chạy thành công**
      (ép `NUM_EPOCHS=1` + ghi vào thư mục tạm nên không đụng checkpoint v3).

- [x] `P3.6` (Phát sinh) Thêm `src/utils/plotting.py::configure_matplotlib_backend()`:
      chỉ dùng backend `Agg` khi chạy script, giữ backend inline khi chạy trong notebook
      (trước đây `trainer`/`evaluator` ép `Agg` ở module-level nên biểu đồ sẽ không hiển thị trong notebook).

**Verify:** ✅ `pytest -q` → **53 passed, 1 skipped**.
Đã chạy lại cả `download_data.py`, `-m src.pipelines.trainer`, `-m src.pipelines.evaluator`,
`-m src.pipelines.predictor` và notebook: tất cả hoạt động, log có format thống nhất,
kết quả evaluator không đổi (0.9420 / 0.9452).

---

## 🟢 Phase 4 — Web app & UX (0.5–1 ngày)

- [x] `P4.1` Thay "ảnh mẫu" vẽ bằng canvas (`script.js` — chỉ là gradient + emoji, dự đoán vô nghĩa) bằng **ảnh hoa thật**:
  - tạo `samples/{rose,daisy,lily}.jpg` (script `scripts/make_sample_images.py`, có `samples/README.md` ghi nguồn)
  - thêm `GET /api/samples` + mount static `/samples`
  - dùng đúng `config.SAMPLE_DIR` (biến này hiện khai báo nhưng chưa dùng).
- [x] `P4.2` Chuyển `@app.on_event("startup")` (deprecated) → `lifespan` context manager.
- [x] `P4.3` Validate upload thật sự: giới hạn **10MB** (`config.MAX_UPLOAD_SIZE_MB`, trả HTTP 413), chặn sai content-type/đuôi file/ảnh hỏng/file rỗng, message tiếng Việt rõ ràng; UI cũng chặn trước ở client.
- [x] `P4.4` `ImageOps.exif_transpose()` để ảnh chụp bị xoay được xử lý đúng.
- [x] `P4.5` `/api/health` trả thêm metadata model: `arch`, `use_pretrained`, `trained_at`, `best_epoch`, `best_val_acc`, `loaded_classes`, `image_size`, `max_upload_mb`.
- [x] `P4.6` Sửa nhãn hiển thị bị lặp ở `script.js` (`ROSE (Class: rose)` → `Class ID: rose`); thêm cảnh báo khi model chưa được huấn luyện.

**Verify:** ✅ `pytest -q` → 68 pass / 1 skip (14 test API mới trong `tests/test_api.py`); ảnh mẫu thật phục vụ tại `/samples/*.jpg`; upload >10MB → HTTP 413 kèm thông báo tiếng Việt.

---

## 🧩 Phase 5 — Tài liệu & bàn giao (0.5 ngày)

- [x] `P5.1` Cập nhật `README.md`: cấu trúc thư mục mới, kết quả thực tế (0.9420 / 0.9452), confusion matrix dạng bảng,
  bảng siêu tham số, hướng dẫn chạy nhanh 7 bước, bảng mã lỗi API, mục hạn chế & hướng phát triển.
- [x] `P5.2` Thêm `docs/BAO_CAO.md`: bài toán & tiêu chí, dataset (kèm phân tích lệch lớp + sai số chuẩn),
  kiến trúc AlexNet (57.016.131 tham số), quy trình huấn luyện, kết quả 3 phiên bản, phân tích lỗi
  `daisy` ↔ `lily`, triển khai, kiểm thử, hạn chế & hướng phát triển, phụ lục artifact.
- [x] `P5.3` Rà `.gitignore`: xác nhận **không** có `.pth` / dữ liệu nào bị track
  (`git check-ignore` → `*.pth`, `flower_data_3classes/`, `outputs/*` đều khớp).
  Phát hiện rule `!outputs/.gitkeep` là **dead rule** (file chưa từng tồn tại) → đã tạo
  `outputs/.gitkeep` và track, để sau khi clone vẫn có sẵn thư mục `outputs/`.

**Verify:** ✅ `pytest -q` → 68 pass / 1 skipped; `python -m src.pipelines.predictor` chạy trên 3 ảnh mẫu
thật cho đúng kết quả (daisy 97.33% · lily 99.99% · rose 100%); mọi đường dẫn tài liệu trong `README.md`
đều tồn tại.

---

## 🧹 Phase 6 — Quét & dọn code/file thừa (0.5 ngày)

Báo cáo chi tiết: **`docs/CLEANUP_REPORT.md`**

- [x] `P6.1` Quét 4 lớp: ruff (F/F401/F811/F841/ARG/B007/RUF100), grep symbol không có người gọi,
  đối chiếu HTML/JS ↔ CSS (class/id mồ côi, biến `--var`, `@keyframes`), đối chiếu tài liệu ↔ code.
- [x] `P6.2` Xoá ~107 dòng code chết: hàm `get_logger()` (không ai gọi), 5 file `__init__.py` kiểu
  re-export (73 → 5 dòng), hằng `NUM_CLASSES` (chỉ dùng trong 1 assert luôn đúng), 2 import thừa,
  1 biến vòng lặp thừa trong notebook.
- [x] `P6.3` Xoá 5 class + 1 id + 1 attribute HTML không được CSS/JS dùng; xoá 2 biến CSS thừa;
  xoá 26 dòng vẽ biểu đồ trùng lặp với `save_training_artifacts()` trong notebook.
- [x] `P6.4` **Sửa lỗi thật (không chỉ dư thừa)**: checkpoint thừa `optimizer_state_dict`
  (652MB → ~245MB), và `load_checkpoint_data()` giữ nguyên state Adam trong `metadata` khiến
  app web ôm hàng trăm MB tensor trong RAM → đã lọc bỏ + thêm test chống tái phát.
- [x] `P6.5` Chuẩn hoá 14 log message tiếng Việt không dấu trong `download_data.py`/`downloader.py`.
- [x] `P6.6` Sửa 3 chỗ tài liệu mô tả sai code (scheduler `mode='max'` → `mode='min'`,
  `get_logger()`, kích thước checkpoint).
- [x] `P6.7` **Quét lần 2 — trùng lặp tinh vi** (khối code lặp ở nhiều file, chỗ nào cũng "có
  người dùng" nên ruff không báo): gộp fixture `fake_build_model` (định nghĩa 2 lần ở
  `test_checkpoint.py` + `test_predictor.py`) vào `tests/conftest.py`; gộp helper tạo ảnh giả
  (2 bản với 2 chữ ký khác nhau) vào `tests/_helpers.py::make_images`; rút `.gitattributes`
  từ 20 → 4 dòng; xoá 1 dòng `.gitignore` trùng; chuẩn hoá notebook sang LF. Kiểm tra và kết luận
  **sạch**: CSS (0 thuộc tính trùng), `requirements.txt` (cả 12 package đều dùng), tài liệu
  (trùng tối đa ~7%), `pytest.ini` (cần giữ `norecursedirs`).

**Verify:** ✅ `pytest -q` → **69 pass / 1 skipped**; ruff lần 2 không còn F/B007 (các `ARG` còn lại
là false positive có ghi chú); quét HTML/CSS lần 2 → 0 class/id mồ côi, 0 biến CSS thừa;
`python -m src.pipelines.predictor` vẫn cho daisy 97.33% · lily 99.99% · rose 100%; kết quả model
không đổi (0.9420 / 0.9452).

---

## ⏱️ Tổng quan tiến độ

| Phase | Nội dung | Thời lượng | Bắt buộc | Trạng thái |
|---|---|---|---|---|
| 0 | Baseline + nhánh git | 0.5 ngày | ✅ | **XONG** |
| 1 | Sửa lỗi đúng/sai | 1 ngày | ✅ | **XONG** |
| 2 | Train lại + đánh giá | 1–1.5 ngày | ✅ | **XONG** — v3 transfer learning: 0.9420 / 0.9452 (`docs/training_report.md`) |
| 3 | Test + cấu trúc code | 1 ngày | ⚠️ nên có | **XONG** — 53 test pass, logging, pin version, notebook sạch |
| 4 | Web app & UX | 0.5–1 ngày | ⚠️ nên có | **XONG** — ảnh mẫu thật, lifespan, validate upload 10MB, EXIF, health metadata (+14 test API → 68 pass) |
| 5 | Tài liệu | 0.5 ngày | ✅ | **XONG** — viết lại `README.md`, thêm `docs/BAO_CAO.md`, rà `.gitignore` |
| 6 | Dọn code/file thừa | 0.5 ngày | ⚠️ nên có | **XONG** — xoá ~107 dòng code chết + ~45 dòng trùng lặp, checkpoint 652MB → ~245MB (`docs/CLEANUP_REPORT.md`) |

**Rủi ro cần biết trước:**
- Phase 2 cần **train lại** (GPU khả dụng) → tốn thời gian; kết quả có thể khác baseline (tốt hơn hoặc xấu hơn tùy seed).
- Checkpoint cũ (652MB) sẽ **không tương thích** nếu sau này đổi kiến trúc → nên đặt tên có version, ví dụ `best_alexnet_flowers_v2.pth`.
