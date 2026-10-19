# Ảnh mẫu cho Web App

Thư mục này chứa 3 ảnh hoa **thật** dùng cho phần "chọn mẫu ảnh thử nghiệm" trên web app.

| File | Lớp | Nguồn |
|------|-----|-------|
| `rose.jpg` | Hoa Hồng (`rose`) | `flower_data_3classes/test/rose/` — ảnh đầu tiên theo thứ tự alphabet |
| `daisy.jpg` | Hoa Cúc (`daisy`) | `flower_data_3classes/test/daisy/` |
| `lily.jpg` | Hoa Ly (`lily`) | `flower_data_3classes/test/lily/` |

## Nguồn gốc & giấy phép

Ảnh được trích từ tập test của bộ dữ liệu Kaggle
[`utkarshsaxenadn/flower-classification-5-classes-roselilyetc`](https://www.kaggle.com/datasets/utkarshsaxenadn/flower-classification-5-classes-roselilyetc)
đã dùng để huấn luyện mô hình trong đồ án. Đây là ảnh **thật**, không phải ảnh vẽ bằng canvas
(như bản cũ) nên kết quả dự đoán trên chúng có ý nghĩa.

Nếu bộ dữ liệu gốc được phát hành với giấy phép hạn chế việc phân phối lại, hãy xoá 3 file này
khỏi Git và tạo lại cục bộ bằng lệnh bên dưới — web app vẫn hoạt động bình thường sau khi có file.

## Tạo lại ảnh mẫu

```bash
python scripts/make_sample_images.py          # chỉ tạo file còn thiếu
python scripts/make_sample_images.py --force  # ghi đè toàn bộ
```

Script thu nhỏ ảnh về tối đa 640px và lưu JPEG (quality 85) để dung lượng repo nhỏ
(tổng ~23KB cho cả 3 ảnh). Ảnh được phục vụ tại `/samples/{rose,daisy,lily}.jpg`
và liệt kê qua `GET /api/samples`.
