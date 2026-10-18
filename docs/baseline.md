# 📌 Baseline — Trạng thái dự án TRƯỚC khi sửa (Phase 0)

Ghi lại ngày: 2026-10-18 · Nhánh: `main` @ `17fdc5b` · Môi trường: torch 2.5.1+cu121 (CUDA available)

## 1. Kết quả đánh giá trên tập TEST (checkpoint hiện có)

Lệnh: `python -m src.pipelines.evaluator`
Log đầy đủ: `outputs/baseline_eval.log`

| Lớp | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| daisy (Hoa Cúc) | 0.82 | 0.87 | 0.84 | 260 |
| lily (Hoa Ly) | 0.87 | 0.79 | 0.83 | 350 |
| rose (Hoa Hồng) | 0.78 | 0.91 | 0.84 | 97 |
| **Accuracy** | | | **0.83** | **707** |
| **Macro avg** | 0.82 | 0.86 | **0.84** | 707 |
| **Weighted avg** | 0.84 | 0.83 | 0.83 | 707 |

**Nhận xét:** lớp `lily` có recall thấp nhất (0.79) và `rose` có precision thấp nhất (0.78) —
tức ảnh lily bị đoán nhầm sang lớp khác nhiều. Tập test bị lệch lớp (rose chỉ 97 ảnh so với 350 của lily)
nên cần ưu tiên chỉ số **macro F1** khi so sánh.

## 2. Metadata checkpoint hiện tại

File: `checkpoints/best_alexnet_flowers.pth` (~684 MB)

| Trường | Giá trị |
|---|---|
| `epoch` | 15 (train đủ 15 epoch, không early stopping) |
| `val_acc` | 0.854 (85.4%) |
| `classes` | `['daisy', 'lily', 'rose']` |
| `model_state_dict` | có |
| `optimizer_state_dict` | có |
| `use_pretrained` | ❌ **KHÔNG LƯU** → rủi ro khi đổi `config.USE_PRETRAINED` |
| `arch` | ❌ KHÔNG LƯU |
| `trained_at` | ❌ KHÔNG LƯU |

## 3. Các vấn đề ghi nhận được ở baseline

1. `torch.load` phát `FutureWarning` (`weights_only` sẽ đổi mặc định thành `True` ở torch ≥ 2.6) → checkpoint dạng dict sẽ không load được.
2. Checkpoint không lưu kiến trúc → `predictor.py` / `evaluator.py` build model theo `config.USE_PRETRAINED`, lệch là load lỗi.
3. `predictor.py` hardcode thứ tự lớp từ `config.TARGET_CLASSES` thay vì đọc từ checkpoint.
4. Chạy `evaluator` bằng `-m` in ra `RuntimeWarning` của runpy: `'src.pipelines.evaluator' found in sys.modules ... prior to execution`
   → do `src/__init__.py` import sẵn `evaluate_model` từ chính module đó (import vòng).

## 4. Tiêu chí nghiệm thu cho Phase 2 (sau khi train lại)

- [ ] Accuracy ≥ 0.83 · Macro F1 ≥ 0.84
- [ ] Recall của `lily` cải thiện so với 0.79 (đây là điểm yếu rõ nhất)
- [ ] Checkpoint mới lưu được `use_pretrained`, `arch`, `classes`, `trained_at`
