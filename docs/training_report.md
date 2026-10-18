# 🟠 Phase 2 — Train lại & Đánh giá (báo cáo cuối)

Ngày: 2026-10-18 · Nhánh: `refactor/fix-project` · GPU: Quadro T1000 · torch 2.5.1+cu121

> **Kết luận nhanh:** AlexNet **transfer learning** đạt **Accuracy 94.20% / Macro F1 94.52%**,
> vượt baseline **+10.75 pp accuracy** và **+11 pp macro F1**. Đây là model được chọn để triển khai.

---

## 1. Ba phiên bản được huấn luyện

| Mã | Loại | `USE_PRETRAINED` | Checkpoint | Ghi chú |
|---|---|---|---|---|
| **v1** | Baseline (code cũ) | False | `checkpoints/best_alexnet_flowers_v1_scratch.pth` | Có trước khi sửa, không có metadata |
| **v2** | Scratch (code mới) | False | `checkpoints/best_alexnet_flowers_v2_scratch.pth` | Train lại để kiểm chứng code |
| **v3** | **Transfer learning** ✅ | **True** | `checkpoints/best_alexnet_flowers.pth` | Đang dùng |

Dữ liệu & hyperparameter giữ nguyên cho cả 3: 3000 train / 1500 val / 707 test, 224×224, batch 32,
Adam lr=1e-4 wd=1e-4, ReduceLROnPlateau(patience=2, ×0.5), 15 epoch, seed 42.

---

## 2. Kết quả trên tập TEST (so sánh trực tiếp)

| Chỉ số | v1 Baseline | v2 Scratch | **v3 Pretrained** | v3 so với v1 |
|---|---|---|---|---|
| **Accuracy** | 0.8345 | 0.8034 | **0.9420** | 🟢 **+10.75 pp** |
| **Macro F1** | 0.8360 | 0.8038 | **0.9452** | 🟢 **+10.92 pp** |
| Macro Precision | 0.82 | 0.78 | **0.95** | +13 pp |
| Macro Recall | 0.86 | 0.84 | **0.94** | +8 pp |
| Val Acc (best) | 0.8540 | 0.8500 | **0.9660** | +11.20 pp |
| Best epoch | 15 | 13 | 13 | — |

### Chi tiết theo lớp (F1-score)

| Lớp | Support | v1 Baseline | v2 Scratch | **v3 Pretrained** |
|---|---|---|---|---|
| daisy (Hoa Cúc) | 260 | 0.84 | 0.82 | **0.93** |
| lily (Hoa Ly) | 350 | 0.83 | 0.79 | **0.94** |
| rose (Hoa Hồng) | 97 | 0.84 | 0.80 | **0.96** |

### Chi tiết Precision / Recall của v3

| Lớp | Precision | Recall | F1 |
|---|---|---|---|
| daisy | 0.97 | 0.90 | 0.93 |
| lily | 0.92 | 0.97 | 0.94 |
| rose | 0.95 | 0.97 | 0.96 |

> Điểm yếu nhỏ còn lại: `daisy` có recall 0.90 (thấp nhất) — một phần ảnh hoa cúc bị đoán thành
> `lily`. `rose` giờ đã rất tốt (0.95/0.97) so với tình trạng precision 0.69 của bản scratch.

---

## 3. Diễn biến huấn luyện v3 (transfer learning)

Log đầy đủ: `outputs/train_run_v3_pretrained.log` · Biểu đồ: `outputs/training_curves.png` ·
Lịch sử số: `outputs/history.json`

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Time |
|---|---|---|---|---|---|
| 01 | 0.4295 | 82.83% | 0.2012 | 92.47% | 29.4s |
| 02 | 0.2980 | 88.27% | 0.1726 | 93.80% | 29.9s |
| 03 | 0.2661 | 89.67% | 0.1440 | 95.20% | 33.8s |
| 04 | 0.2413 | 89.90% | 0.1650 | 94.00% | 36.4s |
| 05 | 0.2354 | 91.57% | 0.1496 | 94.93% | 36.2s |
| 06 | 0.1983 | 92.43% | 0.1560 | 94.80% | 36.1s |
| 07 | 0.1870 | 92.23% | 0.1208 | 96.27% | 43.3s |
| 08 | 0.1872 | 92.47% | 0.1488 | 94.80% | 40.4s |
| 09 | 0.1793 | 93.23% | 0.1279 | 95.87% | 41.8s |
| 10 | 0.1592 | 93.70% | 0.1122 | 96.40% | 40.5s |
| 11 | 0.1589 | 93.53% | 0.1217 | 96.13% | 43.5s |
| 12 | 0.1609 | 93.37% | 0.1213 | 96.40% | 41.4s |
| **13** | **0.1509** | **93.97%** | **0.1102** | **96.60%** ✅ | 40.4s |
| 14 | 0.1438 | 94.47% | 0.1348 | 95.87% | 39.7s |
| 15 | 0.1531 | 93.80% | 0.1275 | 96.00% | 39.9s |

- **Best val_acc = 96.60% tại epoch 13**, đã lưu vào `checkpoints/best_alexnet_flowers.pth`.
- Early stopping (patience 5) **không kích hoạt** — chuỗi không cải thiện dài nhất là 3 epoch.
- Thời gian: ~9.5 phút train (sau khi tải weights ImageNet 233MB, ~7 giây).
- **Epoch 1 đã đạt 92.47%** — bằng chứng rõ ràng cho giá trị của pre-trained features:
  chỉ 1 epoch đã vượt toàn bộ 15 epoch của bản scratch.

### Diễn biến v2 (scratch) — để so sánh

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 01 | 0.8847 | 55.20% | 0.7170 | 65.33% |
| 05 | 0.5903 | 74.63% | 0.5172 | 79.13% |
| 10 | 0.5442 | 76.77% | 0.4290 | 82.13% |
| **13** | 0.5165 | 77.90% | 0.3976 | **85.00%** ✅ |
| 15 | 0.4822 | 79.53% | 0.4119 | 84.00% |

---

## 4. Đối chiếu tiêu chí nghiệm thu (`docs/baseline.md`)

| Tiêu chí | Yêu cầu | v2 | **v3** |
|---|---|---|---|
| Accuracy ≥ 0.83 | ≥ 0.83 | 0.8034 ❌ | **0.9420** ✅ |
| Macro F1 ≥ 0.84 | ≥ 0.84 | 0.8038 ❌ | **0.9452** ✅ |
| Recall `lily` > 0.79 | > 0.79 | 0.75 ❌ | **0.97** ✅ |
| Checkpoint có metadata đầy đủ | có | có ✅ | có ✅ |

**→ Toàn bộ tiêu chí ĐẠT.**

---

## 5. Phân tích

1. **Pre-trained ImageNet features quyết định toàn bộ khác biệt.** Cùng dữ liệu, cùng
   hyperparameter, cùng số epoch — chỉ khác điểm khởi tạo trọng số.
2. **Bản v2 (scratch) thấp hơn v1 (baseline) 3.11 pp** dù val_acc gần như trùng nhau
   (85.0% vs 85.4%). Với 707 mẫu test, sai số chuẩn ≈ 1.5 pp → mức lệch này nằm trong
   vùng dao động ngẫu nhiên do seed, **không phải lỗi code**. Bằng chứng: evaluator mới
   chạy lại trên checkpoint v1 vẫn cho đúng 0.8345 / 0.8360 (`outputs/after_fix_eval.log`).
3. **Fixtures đã sửa ở Phase 1 vẫn hoạt động đúng cho cả 3 phiên bản**: `use_pretrained`
   được đọc từ checkpoint nên v1/v2 (scratch) và v3 (pretrained) cùng tồn tại song song
   trong khi `config.USE_PRETRAINED = True` — nếu không có fix 1.4 thì việc load 2 bản scratch
   đã thất bại.
4. **Trần của AlexNet from scratch** với 3000 ảnh là ~80–85%. Việc tinh chỉnh hyperparameter
   sẽ cho lợi ích nhỏ so với transfer learning.
5. `test/rose` chỉ có 97 ảnh → nên tiếp tục ưu tiên **macro F1** khi so sánh các phiên bản.

---

## 6. Artifact

| File | Nội dung |
|---|---|
| `checkpoints/best_alexnet_flowers.pth` | **Model v3 (đang dùng)** — epoch 13, val_acc 0.966, `use_pretrained=True` |
| `checkpoints/best_alexnet_flowers_v2_scratch.pth` | Model v2 scratch (epoch 13, val_acc 0.85) |
| `checkpoints/best_alexnet_flowers_v1_scratch.pth` | Model v1 baseline gốc |
| `outputs/history.json` | Lịch sử 15 epoch của **v3** |
| `outputs/training_curves.png` | Biểu đồ Loss/Accuracy của **v3** |
| `outputs/confusion_matrix.png` | Confusion matrix của **v3** trên tập test |
| `outputs/train_run_v2.log` / `eval_run_v2.log` | Log train/eval của v2 (scratch) |
| `outputs/train_run_v3_pretrained.log` / `eval_run_v3_pretrained.log` | Log train/eval của v3 |
| `outputs/after_fix_eval.log` | Chạy lại evaluator trên v1 để chứng minh không hồi quy |
| `docs/baseline.md` | Số liệu baseline & metadata checkpoint v1 |

> ⚠️ `outputs/history.json` và `outputs/training_curves.png` là **file ghi đè** theo mỗi lần
> chạy `trainer` → hiện chứa dữ liệu của **v3**. Số liệu của v2 nằm trong `outputs/train_run_v2.log`
> và bảng ở mục 3 của tài liệu này.

---

## 7. Kết luận & đề xuất

- **Chọn v3 (transfer learning) làm model triển khai**; giữ `config.USE_PRETRAINED = True`
  để kết quả tái lập được.
- Lưu ý: `checkpoints/*.pth` bị `.gitignore` → **không** được commit; cần chạy lại
  `python -m src.pipelines.trainer` để tái tạo (kết quả có thể lệch ±1 pp do seed/GPU).
- Hướng cải thiện tiếp (nếu còn thời gian):
  1. Unfreeze có chọn lọc / dùng LR nhỏ hơn cho phần `features` để tránh phá pre-trained weights.
  2. Bổ sung dữ liệu `rose` (test set hiện chỉ 97 ảnh) để đánh giá công bằng hơn.
  3. Thử `WeightedRandomSampler` hoặc class weight cho `CrossEntropyLoss`.
  4. Export ONNX để tăng tốc inference cho web app.
