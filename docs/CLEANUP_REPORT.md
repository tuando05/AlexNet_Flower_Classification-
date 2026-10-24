# 🧹 BÁO CÁO QUÉT & DỌN DẸP CODE / FILE THỪA

Ngày: 2026-10-19 · Nhánh: `refactor/fix-project` · Phạm vi: **toàn bộ repo** (trừ `.venv`, dữ liệu, checkpoint)

> **Kết luận ngắn:** đã xoá **~107 dòng code chết** (lần quét 1) và **~45 dòng code trùng lặp**
> (lần quét 2), 5 class/id HTML không dùng, gộp 5 file `__init__.py` từ "re-export" xuống docstring,
> và **phát hiện 1 vấn đề thật** (không chỉ là dư thừa): mỗi checkpoint lưu thừa
> `optimizer_state_dict` → file **652MB thay vì ~218MB**, đồng thời app web giữ hàng trăm MB tensor đó
> trong RAM suốt vòng đời tiến trình. Không đổi kết quả model (vẫn 0.9420 / 0.9452), bộ test
> **69 passed / 1 skipped**.

---

## 1. Phương pháp quét

Quét bằng 4 lớp kiểm tra độc lập, không chỉ đọc bằng mắt:

| Lớp | Công cụ | Bắt được |
|---|---|---|
| 1. Phân tích tĩnh Python | `ruff check . --select F,F401,F811,F841,ARG,B007,B018,RUF100` | import thừa, biến/hàm thừa, tham số thừa, biến vòng lặp không dùng |
| 2. Tìm symbol không có người gọi | `grep` toàn repo theo tên hàm/biến | hàm/hằng số được định nghĩa nhưng không nơi nào gọi |
| 3. Đối chiếu HTML/JS ↔ CSS | script tự viết | class/id trong HTML không có rule CSS và không được JS tham chiếu; biến CSS `--var` khai báo nhưng không dùng; `@keyframes` không dùng |
| 4. Đối chiếu tài liệu ↔ code | đọc `README.md`, `docs/*.md` với code | tài liệu mô tả sai so với code (đã sửa, xem mục 6) |

Các lệnh cụ thể ở **mục 9 (phụ lục)** để có thể chạy lại bất cứ lúc nào.

---

## 2. Tóm tắt kết quả (lần quét 1)

| Hạng mục | Trước | Sau |
|---|---|---|
| Dòng code (23 file bị sửa) | — | **−146 / +124** (phần thêm là docstring giải thích, 1 test mới, tài liệu) |
| Code chết thật sự bị xoá | 107 dòng | **0** |
| `__init__.py` kiểu "re-export" | 5 file (73 dòng) | 5 file docstring (**5 dòng**) |
| Hàm không có người gọi | 1 (`get_logger`) | **0** |
| Hằng số không dùng | 1 (`NUM_CLASSES`) | **0** |
| Import không dùng | 2 | **0** |
| Class/id HTML không dùng | 5 class + 1 id + 1 attribute | **0** |
| Biến CSS không dùng | 2 | **0** |
| Dòng code trùng lặp trong notebook | 26 | **0** |
| Kích thước checkpoint (mỗi bản) | 652 MB | **217,5 MB** (đã bỏ state Adam ở **cả 3 file**) |
| RSS của app khi giữ metadata | giữ nguyên state Adam (~400MB) | **chỉ 10 key nhỏ** |
| Test | 68 passed | **69 passed / 1 skipped** |

---

## 3. Chi tiết đã xoá — Python

### 3.1 Hàm `get_logger()` — `src/utils/logger.py`
- **Xoá 5 dòng.** Hàm chỉ là vỏ bọc của `logging.getLogger(name)` và **không nơi nào gọi**
  (`grep get_logger` chỉ ra: định nghĩa + 1 dòng trong `src/utils/__init__.py` + tài liệu).
- **Xác minh:** sau khi xoá, `grep` toàn repo không còn kết quả trong code; test vẫn pass.
- **Giữ lại:** `setup_logging()` (được gọi ở mọi entry point) và các hằng `LOG_FORMAT`,
  `DATE_FORMAT`, `DEFAULT_LEVEL`.

### 3.2 Năm file `__init__.py` kiểu "re-export" — 73 dòng → 5 dòng
`src/__init__.py`, `src/pipelines/__init__.py`, `src/models/__init__.py`, `src/data/__init__.py`,
`src/utils/__init__.py` đều dựng cơ chế export (`__all__` + lazy import PEP 562 hoặc import trực tiếp)
nhưng **không tồn tại câu lệnh `from src import ...` / `from src.pipelines import ...` nào trong repo**.

| File | Cơ chế cũ | Vấn đề |
|---|---|---|
| `src/__init__.py` | `__getattr__` lazy cho 5 symbol | 29 dòng không ai dùng |
| `src/pipelines/__init__.py` | `__getattr__` lazy cho 3 symbol | 20 dòng không ai dùng |
| `src/data/__init__.py` | import trực tiếp 5 symbol | 11 dòng; **kéo theo cả `kagglehub`/`torchvision`** mỗi lần `import src.data` |
| `src/models/__init__.py` | import trực tiếp alexnet + builder | 3 dòng; **kéo theo torch + torchvision** |
| `src/utils/__init__.py` | import trực tiếp 5 symbol | 5 dòng; kéo theo matplotlib |

→ Thay bằng docstring ngắn giải thích **vì sao cố ý để trống**. Lợi ích phụ: `import src.models`
không còn nạp sẵn torch/torchvision/matplotlib, và không còn nguy cơ tái phát `RuntimeWarning`
của runpy khi chạy `python -m src.pipelines.<module>`.

**Xác minh:** `from src.models import builder`, `from src.data import downloader`,
`from src.pipelines import predictor` (các import thật trong test) vẫn hoạt động — 69 test pass.

### 3.3 Hằng số `NUM_CLASSES` — `config.py`
- **Xoá 1 dòng.** Cách dùng duy nhất là một test tự so với chính định nghĩa của nó:
  `assert config.NUM_CLASSES == len(config.TARGET_CLASSES)` — khẳng định này **luôn đúng** và không
  kiểm tra được gì về hành vi của app (code thật luôn dùng `len(class_names)` lấy từ checkpoint).
- Kèm theo: xoá luôn dòng assert vô nghĩa đó trong `tests/test_model.py`, giữ 2 assert còn lại
  (`CLASS_NAMES_VI` phủ đủ các lớp, `EARLY_STOPPING_PATIENCE >= 0`).

### 3.4 Import không dùng
| File | Import | Ghi chú |
|---|---|---|
| `tests/_helpers.py` | `import torch` | chỉ dùng `torch.nn` |
| `tests/test_transforms.py` | `import torchvision.transforms as T` | không dùng `T` |

### 3.5 Biến vòng lặp không dùng — notebook
`for name, info in sorted(result['probabilities'].items(), ...)` → `name` không dùng
→ đổi sang `for info in sorted(result['probabilities'].values(), ...)`.

---

## 4. Chi tiết đã xoá — HTML / CSS / Notebook

### 4.1 `app/templates/index.html`
| Đã xoá | Lý do |
|---|---|
| `class="panel"` | không có rule `.panel` trong CSS, không JS nào dùng (`.glass-panel` mới là class được style) |
| `class="upload-panel"`, `class="results-panel"` | không có rule CSS, không dùng ở đâu |
| `class="drop-zone-content"` | dư — JS chỉ dùng `id="dropZoneContent"`; không có rule CSS |
| `class="results-content"` | dư — JS/CSS chỉ dùng `id="resultsContent"` |
| `id="winnerCard"` | không CSS và không JS nào tham chiếu (`.winner-card` mới là class được style) |
| `data-sample="{{ sample.key }}"` | tàn dư của bản cũ: JS giờ đọc `data-url` để tải ảnh mẫu thật |

### 4.2 `app/static/style.css`
- Xoá 2 biến CSS khai báo nhưng không dùng ở đâu: `--primary-hover`, `--accent-amber`.
- **Đã quét toàn bộ 530 dòng CSS:** không còn class/id selector nào "mồ côi", không có
  `@keyframes` thừa. (`.main-grid` xuất hiện 2 lần là **cố ý** — 1 lần base + 1 lần trong
  `@media (max-width: 900px)`.)

### 4.3 `notebooks/alexnet_flower_classification.ipynb` — xoá 26 dòng trùng lặp
- Cell "Trực quan hóa quá trình huấn luyện" **vẽ lại y hệt** biểu đồ mà
  `save_training_artifacts()` (gọi ở cell trên) đã vẽ và lưu thành `outputs/training_curves.png`
  → thay bằng 4 dòng hiển thị chính file PNG đó. Notebook giữ đúng vai trò "mỏng" đã chốt ở Phase 3.
- Cell dự đoán: đường dẫn ảnh cũ tự `sorted(os.listdir(...))` trong tập test → đổi sang dùng
  `samples/rose.jpg` (ảnh mẫu thật, đã commit, luôn tồn tại).

---

## 5. ⚠️ Phát hiện quan trọng: `optimizer_state_dict` trong checkpoint

Đây là hạng mục **không chỉ là "dư thừa" mà là lỗi về dung lượng và bộ nhớ**.

### 5.1 Vấn đề
`trainer.py` lưu checkpoint mỗi khi `val_acc` tốt hơn, và lưu kèm `optimizer.state_dict()`:

```python
torch.save({
    'model_state_dict': model.state_dict(),          # ~228 MB (57M tham số × 4 byte)
    'optimizer_state_dict': optimizer.state_dict(),  # ~456 MB (Adam: 2 moment/param)
    ...
})
```

- Dự án **không hỗ trợ resume training** — không có dòng code nào đọc lại `optimizer_state_dict`.
- Nhưng `load_checkpoint_data()` lại đưa **toàn bộ** các key còn lại vào `metadata`, kể cả key này.
  `FlowerPredictor` giữ `self.metadata` suốt vòng đời → **app web giữ nguyên ~400MB tensor trong RAM**
  chỉ để rồi không bao giờ đọc tới. Đã quan sát trực tiếp: in `metadata` sau khi load ra một dict
  khổng lồ chứa `exp_avg`/`exp_avg_sq` của cả 16 tensor nhóm.

### 5.2 Đã sửa
| File | Thay đổi |
|---|---|
| `src/pipelines/trainer.py` | Không lưu `optimizer_state_dict` nữa (kèm comment giải thích) |
| `src/models/builder.py` | Thêm `_NON_METADATA_KEYS = ('model_state_dict', 'optimizer_state_dict')` và lọc bỏ khỏi `metadata` |
| `tests/test_checkpoint.py` | Thêm test `test_load_checkpoint_data_excludes_optimizer_state` để không tái phát |

### 5.3 Kết quả đo trên checkpoint thật (652MB, bản cũ vẫn còn nguyên key)
```
metadata keys: ['arch', 'batch_size', 'classes', 'epoch', 'image_size',
                'learning_rate', 'num_classes', 'trained_at', 'use_pretrained', 'val_acc']
co optimizer_state_dict: False
val_acc: 0.966 | arch: alexnet | trained_at: 2026-10-18T22:12:08
```
- Trước: `metadata` chứa cả state Adam. **Sau: chỉ 10 key nhỏ**, model vẫn load và dự đoán đúng
  như cũ (daisy 97.33% · lily 99.99% · rose 100.00%).
- **Checkpoint train mới sẽ chỉ còn ~218MB** (giảm ~67%).
- Checkpoint cũ vẫn load bình thường vì `model_state_dict` không đổi → nên có thể **dọn luôn
  state Adam bên trong 3 file đang có** (xem 5.4).

### 5.4 Dọn state Adam trong 3 checkpoint đang có (đã thực hiện)
Bỏ key `optimizer_state_dict` khỏi cả 3 file, **giữ nguyên `model_state_dict`**. Ghi an toàn: lưu ra
`.tmp` → kiểm tra file tạm load lại được + số tensor khớp → `os.replace()` (atomic) mới thay file gốc.

| File | Trước | Sau | Tiết kiệm |
|---|---|---|---|
| `best_alexnet_flowers.pth` (đang triển khai) | 652,5 MB | **217,5 MB** | 435,0 MB |
| `best_alexnet_flowers_v1_scratch.pth` | 652,5 MB | **217,5 MB** | 435,0 MB |
| `best_alexnet_flowers_v2_scratch.pth` | 652,5 MB | **217,5 MB** | 435,0 MB |
| **Tổng** | 1957,5 MB | **652,5 MB** | **~1,28 GB** |

**Kiểm chứng sau khi dọn** (`model_state_dict` 16 tensor, không mất mát gì):
- `python -m src.pipelines.predictor` → daisy 97.33% · lily 99.99% · rose 100.00% (y hệt trước).
- `python -m src.pipelines.evaluator` → **Accuracy 0.9420 | Macro F1 0.9452** — **trùng khớp tuyệt đối**
  với kết quả trước khi dọn (mục 5).

---

## 6. Tài liệu mô tả sai so với code (đã sửa)

Quét lớp 4 phát hiện 3 chỗ tài liệu không khớp code — đã sửa để báo cáo không còn sai:

| Tài liệu | Trước | Sau (đúng với code) |
|---|---|---|
| `docs/BAO_CAO.md` | `ReduceLROnPlateau(mode='max', ...)` "giảm LR khi **val_acc** chững lại" | `mode='min'` theo **val_loss** (`trainer.py`) |
| `docs/BAO_CAO.md`, `README.md` | "thư viện trong `src/` chỉ gọi `get_logger()`" | `logging.getLogger(__name__)` — và giải thích thêm vì sao entry point chạy bằng `-m` đặt tên logger cố định (để log hiện `src.pipelines.trainer` thay vì `__main__`) |
| `README.md`, `docs/BAO_CAO.md` | checkpoint "~684MB/phiên bản" | 652MB cho bản cũ, **~218MB** sau khi bỏ state Adam |

---

## 7. Chuẩn hoá nhất quán: log tiếng Việt

`src/data/downloader.py` và `download_data.py` còn 14 thông báo viết **không dấu**
(`"Du lieu da ton tai tai"`, `"Khong tim thay thu vien 'kagglehub'"`, banner `"KICH BAN TAI..."`).
Đây là di sản của bản vá lỗi `UnicodeEncodeError` thời chưa có `src/utils/console.py`. Vì mọi entry
point nay đã gọi `setup_console()` (ép stdout/stderr sang UTF-8), các thông báo này đã được
viết lại có dấu, thống nhất với phần còn lại của dự án.

**Xác minh:** chạy lại `python download_data.py` → in đúng tiếng Việt có dấu, không lỗi encoding.

---

## 8. Những thứ CỐ Ý GIỮ (đã cân nhắc và loại trừ)

Ghi lại để chứng minh đã xét chứ không bỏ sót:

| Hạng mục | Vì sao KHÔNG xoá |
|---|---|
| `BASE_DIR` + `sys.path.insert` ở 6 entry point | Cần thiết khi chạy file trực tiếp hoặc từ thư mục khác (đặc biệt `scripts/make_sample_images.py` nằm trong thư mục con). Xoá đi sẽ dễ vỡ khi người dùng chạy `python scripts\...` |
| `src/data/dataset.py::prepare_flower_dataset` | **Có người gọi thật**: `create_dataloaders()` gọi nó ở mọi lần train/eval. Nhánh `source_path` (tự chia 80/10/10) và nhánh Kaggle đều còn giá trị |
| `src/data/downloader.py::_find_v1_root` (hàm "private") | Được gọi bởi `download_and_filter_dataset` **và** được test riêng (structure Kaggle) |
| `docs/baseline.md`, `docs/training_report.md`, `REFACTOR_PLAN.md` | Là **hồ sơ kiểm toán** (audit trail) của 6 giai đoạn: số liệu baseline, so sánh 3 phiên bản, lý do từng quyết định. Có trùng lặp nội dung với `BAO_CAO.md` nhưng phục vụ mục đích khác (lịch sử vs tổng hợp) |
| `samples/*.jpg` + `samples/README.md` | 23KB, cần cho tính năng "ảnh mẫu" của web app |
| `download_data.py` (bản vá `.venv/Lib/site-packages`) | Chủ đích: cho phép chạy bằng Python hệ thống (không kích hoạt venv) |
| `outputs/*.log`, `outputs/*.png`, `outputs/history.json` | Bằng chứng số cho `docs/training_report.md` (đều bị `.gitignore`) |
| Cảnh báo `ARG001/ARG002` còn lại của ruff | **False positive**: `lifespan(app)` theo giao ước ASGI của Starlette; `FakePredictor.predict_image(image)` phải khớp interface; `fake_build_model(use_pretrained=...)` phải khớp chữ ký `build_model`; các tham số fixture của pytest do pytest tiêm vào |
| `.gitignore` boilerplate (`env/`, `*.so`, `Thumbs.db`...) | Quy ước chuẩn của Python/git, giữ để repo dùng lại được ở máy khác |
| `notebooks/*.ipynb` (11 cell code) | Vẫn là 1 trong các sản phẩm bàn giao; đã làm mỏng, không xoá |
| `.gitattributes` | Đảm bảo LF thống nhất trên Windows |

**Đã thêm** (không phải xoá): `.ruff_cache/`, `.mypy_cache/` vào `.gitignore`;
`ruff==0.17.1` vào `requirements-dev.txt` để có thể chạy lại chính xác các lệnh quét ở mục 9;
1 test mới cho việc lọc `optimizer_state_dict`.

---

## 9. Kiểm chứng sau khi dọn

| Kiểm tra | Lệnh | Kết quả |
|---|---|---|
| Toàn bộ test | `pytest -q` | **69 passed, 1 skipped** (~8s) |
| Rà code chết lần 2 | `ruff check . --select F,B007,ARG` | Không còn F/B007; các `ARG` còn lại đều là false positive (mục 8) |
| Rà HTML/CSS lần 2 | script đối chiếu | 0 class/id mồ côi, 0 biến CSS thừa |
| CLI dữ liệu | `python download_data.py` | In tiếng Việt có dấu, dữ liệu đã có → không tải lại |
| CLI dự đoán | `python -m src.pipelines.predictor` | daisy 97.33% · lily 99.99% · rose 100.00% (không đổi) |
| Load checkpoint thật | script đọc `metadata` | 10 key nhỏ, **không có** `optimizer_state_dict` |
| Web app | `pytest tests/test_api.py` | 15 test API pass, `/` render ảnh mẫu thật |
| Kết quả model | evaluator | **0.9420 / 0.9452** — không đổi |

---

## 10. File lớn: đã xử lý phần nào, còn lại gì

**Đã xử lý** (bạn chọn phương án 1, xem mục 5.4): bỏ state Adam khỏi cả 3 checkpoint
→ `1957,5 MB` còn `652,5 MB`, **tiết kiệm ~1,28 GB**, kết quả model y hệt (0.9420 / 0.9452).

**Còn lại**:

| File | Dung lượng | Ghi chú |
|---|---|---|
| `checkpoints/best_alexnet_flowers.pth` | 217,5 MB | **Model đang triển khai** — cần giữ |
| `outputs/*.log` (6 file) | ~32 KB | Bằng chứng số cho báo cáo — nhỏ, nên giữ |
| `.pytest_cache/`, `.ruff_cache/`, `__pycache__/` | vài MB | Cache, đã thêm vào `.gitignore`; xoá lúc nào cũng được (tự sinh lại) |

**Tiếp theo (2026-10-24):** 2 checkpoint đối chứng v1/v2 đã được **xoá** (mỗi file 217,5 MB, tiết
kiệm thêm ~435 MB). Số liệu đối chứng vẫn còn trong `docs/baseline.md` và
`docs/training_report.md`; chỉ mất khả năng chạy lại `evaluator` trên chúng.
**Tổng dung lượng checkpoint hiện tại: 217,5 MB** (chỉ còn model v3 đang triển khai).

---

## 11. Quét lần 2 — các dạng trùng lặp "tinh vi"

Sau lần 1 (đã xoá code/hàm/hằng **không có người gọi**), lần 2 nhắm vào dạng khó thấy hơn:
**cùng một khối code/logic bị viết lặp ở nhiều chỗ** (mỗi bản đều "có người dùng" nên công cụ
thông thường không báo).

### 11.1 Phương pháp
| Phép kiểm | Cách làm |
|---|---|
| CSS khai báo trùng thuộc tính trong cùng một rule | tách selector/body bằng regex, đếm property trùng |
| Khối code 5 dòng giống nhau ở ≥ 2 file `.py` | chuẩn hoá từng dòng (bỏ comment/dòng trắng), trượt cửa sổ 5 dòng |
| Fixture / helper định nghĩa lặp trong `tests/` | `grep` tên fixture/helper trên toàn thư mục test |
| Tài liệu lặp | đếm tỉ lệ dòng chung giữa mọi cặp file `.md` |
| Phụ thuộc không dùng | đối chiếu từng package trong `requirements.txt` với `grep` |

### 11.2 Đã sửa

| Hạng mục | Trước | Sau |
|---|---|---|
| Fixture `fake_build_model` | định nghĩa **2 lần** (`test_checkpoint.py::patched_builder`, `test_predictor.py::tiny_model_builder`) | **1 lần** trong `tests/conftest.py` (`fake_model_builder`) |
| Helper tạo ảnh giả | định nghĩa **2 lần** với 2 chữ ký khác nhau (`_make_images(dir, names)` và `_make_images(dir, count)`) | **1 lần** trong `tests/_helpers.py` (`make_images(dir, *names)`) |
| `.gitattributes` | 20 dòng: liệt kê 9 phần mở rộng text + 5 phần mở rộng binary | **4 dòng** — `* text=auto eol=lf` đã bao quát mọi file, git tự nhận diện nhị phân |
| `.gitignore` | 2 dòng cho `.ipynb_checkpoints/` (dòng `notebooks/.ipynb_checkpoints/` là dư) | 1 dòng |
| `notebooks/*.ipynb` | working tree bị CRLF (git phải cảnh báo + tự đổi sang LF mỗi lần `git add`) | đã chuẩn hoá **LF** |

→ Xoá thêm **~45 dòng** trùng lặp. Sau khi gộp, sửa `fake_build_model` chỉ cần sửa **1 chỗ** cho
cả 2 file test (trước đây sửa 1 bên rất dễ quên bên kia).

### 11.3 Đã kiểm tra và **kết luận là sạch** (không cần sửa)

| Hạng mục | Kết quả kiểm tra |
|---|---|
| CSS | **0** rule có thuộc tính khai báo trùng; 0 selector mồ côi (đã quét cả `@media`, `@keyframes`) |
| `requirements.txt` | **cả 12 package đều được import thật**: torch, torchvision, pillow, numpy, matplotlib, seaborn, scikit-learn, fastapi, uvicorn, python-multipart (bắt buộc cho `UploadFile`), Jinja2 (cho `Jinja2Templates`), kagglehub |
| Tài liệu | mức trùng cao nhất giữa 2 file `.md` chỉ **~7%** (README ↔ BAO_CAO) → **không** có chuyện "3 file chép lại một bảng kết quả" |
| `pytest.ini` | `norecursedirs` thoạt nhìn dư (vì đã có `testpaths = tests`) nhưng **cần giữ**: khi ai đó chạy `pytest .` thì `testpaths` bị bỏ qua, lúc đó `norecursedirs` mới chặn `.venv`/`flower_data_3classes` |
| Notebook (cell 4 và cell 7) | cùng gọi bước chuẩn bị dữ liệu là **cố ý sư phạm** (giải thích từng bước), cell 7 dùng lại cache nên không tốn thời gian |
| `scripts/make_sample_images.py`, `download_data.py` | hai entry point khác nhau, không trùng logic (logic tải đã gom về `src/data/downloader.py` từ Phase 1) |

### 11.4 Còn lặp nhưng **cố ý giữ**
| Vị trí | Số dòng | Vì sao không gộp |
|---|---|---|
| Khối bootstrap `BASE_DIR` + `sys.path.insert` | 6 entry point × ~4 dòng | Bản chất "con gà - quả trứng": phải có `sys.path` **trước khi** import được `config`/`src`, nên không thể đưa vào một helper dùng chung. Chỉ gộp được nếu bỏ chế độ chạy trực tiếp file (hiện `scripts/make_sample_images.py` nằm trong thư mục con bắt buộc cần nó) |
| 3 dòng import `setup_console` / `setup_logging` / `configure_matplotlib_backend` ở trainer & evaluator | ~6 dòng | Cùng lý do; đã gom **thân** hàm vào `src/utils/`, chỉ còn phần import là trùng |

---

## 12. Phụ lục — lệnh quét để tái sử dụng

```bash
# 1. Code chết trong Python (import/hàm/biến/tham số thừa)
python -m ruff check . --select F,F401,F811,F841,ARG,B007,RUF100 \
    --exclude flower_data_3classes,.venv,outputs

# 2. Tìm symbol không có người gọi (ví dụ đã dùng để kết luận get_logger là dead)
#    (thay <ten_ham> bằng tên cần kiểm tra)
git grep -n "<ten_ham>"

# 3. Đối chiếu HTML/JS ↔ CSS (script tạm, xem lịch sử phiên làm việc):
#    - class trong HTML không có rule CSS và không được JS tham chiếu
#    - id trong HTML không được JS dùng
#    - biến CSS --var khai báo nhưng không dùng
#    - @keyframes không dùng

# 4. Kiểm tra .gitignore có thật sự chặn file lớn
git check-ignore -v checkpoints/best_alexnet_flowers.pth flower_data_3classes outputs/history.json
git ls-files | Select-String -Pattern '\.pth$|flower_data_3classes'   # phải rỗng

# 5. Kiểm chứng sau khi sửa
python -m pytest -q
python -m src.pipelines.predictor
```

### Lệnh cho lần quét 2 (trùng lặp)

```bash
# a) Fixture/helper định nghĩa lặp trong thư mục test (ví dụ)
#    (thay <ten> bằng tên fixture/helper cần kiểm tra)
git grep -n "def <ten>" -- tests/

# b) Phụ thuộc không dùng: đối chiếu từng dòng trong requirements.txt với code
git grep -n "import <ten_package>"

# c) CSS khai báo trùng thuộc tính trong cùng một rule / biến --var không dùng:
#    tách selector-body bằng regex rồi đếm property trùng (xem mục 11.1)

# d) Tài liệu lặp: đếm tỉ lệ dòng chung giữa mọi cặp file *.md (xem mục 11.1)

# e) Chuẩn hoá line-ending file notebook (tránh git cảnh báo CRLF mỗi lần add)
python -c "from pathlib import Path; p=Path('notebooks/alexnet_flower_classification.ipynb'); p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n'))"
```

