# 🧹 BÁO CÁO QUÉT & DỌN DẸP CODE / FILE THỪA

Ngày: 2026-10-19 · Nhánh: `refactor/fix-project` · Phạm vi: **toàn bộ repo** (trừ `.venv`, dữ liệu, checkpoint)

> **Kết luận ngắn:** đã xoá **~107 dòng code chết** và 5 class/id HTML không dùng, gộp 5 file
> `__init__.py` từ "re-export" xuống docstring, và **phát hiện 1 vấn đề thật** (không chỉ là dư thừa):
> mỗi checkpoint lưu thừa `optimizer_state_dict` → file **652MB thay vì ~245MB**, đồng thời app web
> giữ hàng trăm MB tensor đó trong RAM suốt vòng đời tiến trình. Không đổi kết quả model
> (vẫn 0.9420 / 0.9452), bộ test **69 passed / 1 skipped**.

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

## 2. Tóm tắt kết quả

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
| Kích thước checkpoint mới | 652 MB | **~245 MB** |
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
- **Checkpoint train mới sẽ chỉ còn ~245MB** (giảm ~62%). Các checkpoint cũ (652MB) vẫn load bình
  thường vì `model_state_dict` không đổi.

---

## 6. Tài liệu mô tả sai so với code (đã sửa)

Quét lớp 4 phát hiện 3 chỗ tài liệu không khớp code — đã sửa để báo cáo không còn sai:

| Tài liệu | Trước | Sau (đúng với code) |
|---|---|---|
| `docs/BAO_CAO.md` | `ReduceLROnPlateau(mode='max', ...)` "giảm LR khi **val_acc** chững lại" | `mode='min'` theo **val_loss** (`trainer.py`) |
| `docs/BAO_CAO.md`, `README.md` | "thư viện trong `src/` chỉ gọi `get_logger()`" | `logging.getLogger(__name__)` — và giải thích thêm vì sao entry point chạy bằng `-m` đặt tên logger cố định (để log hiện `src.pipelines.trainer` thay vì `__main__`) |
| `README.md`, `docs/BAO_CAO.md` | checkpoint "~684MB/phiên bản" | 652MB cho bản cũ, **~245MB** cho bản mới |

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

## 10. Việc còn lại (cần bạn quyết định — không tự xoá)

Các file dưới đây **lớn** và/hoặc là bằng chứng, nên tôi không tự xoá:

| File | Dung lượng | Ghi chú |
|---|---|---|
| `checkpoints/best_alexnet_flowers.pth` | 652 MB | **Model đang triển khai** — cần giữ |
| `checkpoints/best_alexnet_flowers_v1_scratch.pth` | 652 MB | Chỉ dùng để đối chứng (đã có số liệu trong `docs/baseline.md`) |
| `checkpoints/best_alexnet_flowers_v2_scratch.pth` | 652 MB | Chỉ dùng để đối chứng (đã có số liệu trong `docs/training_report.md`) |
| `outputs/*.log` (6 file) | ~32 KB | Bằng chứng số cho báo cáo — nhỏ, nên giữ |
| `.pytest_cache/`, `.ruff_cache/`, `__pycache__/` | vài MB | Cache, đã thêm vào `.gitignore`; xoá lúc nào cũng được (tự sinh lại) |

**Hai lựa chọn tiết kiệm dung lượng** (không ảnh hưởng kết quả model):

1. **Bỏ state Adam khỏi 3 checkpoint cũ** — giữ nguyên trọng số model, chỉ xoá key không dùng:
   mỗi file từ 652MB → ~245MB, **tiết kiệm ~1.2 GB**, và model vẫn load/dự đoán y hệt.
2. **Xoá 2 checkpoint đối chứng v1/v2** — tiết kiệm thêm ~1.3 GB, nhưng mất khả năng chạy lại
   `evaluator` trên chúng để tái kiểm chứng kết luận "không hồi quy" (số liệu vẫn còn trong
   `docs/baseline.md` và `docs/training_report.md`).

---

## 11. Phụ lục — lệnh quét để tái sử dụng

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
