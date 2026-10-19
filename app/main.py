import io
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps, UnidentifiedImageError

# Ensure root directory is in python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from src.pipelines.predictor import FlowerPredictor
from src.utils.console import setup_console
from src.utils.logger import setup_logging

setup_console()
setup_logging()

logger = logging.getLogger('app.main')

# --- Đường dẫn tài nguyên ---
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(config.SAMPLE_DIR, exist_ok=True)

MAX_UPLOAD_BYTES = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Predictor toàn cục, được khởi tạo trong lifespan
predictor = None


def create_predictor():
    """
    Tạo FlowerPredictor.

    Tách riêng thành hàm để test có thể mock, tránh phải load checkpoint ~240MB.
    """
    return FlowerPredictor(model_path=config.MODEL_PATH, device=config.DEVICE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo model khi server start, giải phóng khi tắt (thay cho on_event)."""
    global predictor

    logger.info("Đang khởi tạo FlowerPredictor...")
    predictor = create_predictor()

    if predictor.is_trained:
        logger.info(
            "Khởi tạo thành công | classes=%s | arch=%s | val_acc=%s | use_pretrained=%s",
            predictor.class_names,
            predictor.metadata.get('arch'),
            predictor.metadata.get('val_acc'),
            predictor.metadata.get('use_pretrained'),
        )
    else:
        logger.warning(
            "CHƯA có checkpoint tại '%s'. Server vẫn chạy nhưng kết quả dự đoán sẽ vô nghĩa. "
            "Hãy chạy: python -m src.pipelines.trainer",
            config.MODEL_PATH,
        )

    yield

    predictor = None
    logger.info("Đã giải phóng FlowerPredictor.")


app = FastAPI(
    title="Phân Loại Ảnh Hoa AlexNet API",
    description="REST API và Web App phân loại 3 loại hoa (Hoa Hồng, Hoa Cúc, Hoa Ly) bằng mô hình Deep Learning AlexNet",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/samples", StaticFiles(directory=config.SAMPLE_DIR), name="samples")

templates = Jinja2Templates(directory=TEMPLATES_DIR)


def get_sample_images():
    """
    Danh sách ảnh mẫu thật có sẵn trong `config.SAMPLE_DIR`.

    Ảnh được tạo bằng `python scripts/make_sample_images.py`.
    Trả về list rỗng nếu chưa có -> UI sẽ tự ẩn phần ảnh mẫu.
    """
    samples = []
    for cls in config.TARGET_CLASSES:
        for ext in ('.jpg', '.jpeg', '.png'):
            filename = f"{cls}{ext}"
            if os.path.exists(os.path.join(config.SAMPLE_DIR, filename)):
                samples.append({
                    'key': cls,
                    'label': config.CLASS_NAMES_VI.get(cls, cls),
                    'filename': filename,
                    'url': f"/samples/{filename}",
                })
                break
    return samples


def _open_uploaded_image(contents):
    """
    Chuyển bytes upload thành ảnh RGB.

    - Sửa hướng ảnh theo EXIF (ảnh chụp từ điện thoại thường bị xoay).
    - Báo lỗi 400 rõ ràng nếu file không phải ảnh hoặc bị hỏng.
    """
    try:
        image = Image.open(io.BytesIO(contents))
        image = ImageOps.exif_transpose(image) or image
        return image.convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Không đọc được nội dung ảnh. File có thể bị hỏng hoặc không phải là ảnh.",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể xử lý ảnh: {e}")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Renders the main modern web UI dashboard.
    """
    context = {
        "classes": config.TARGET_CLASSES,
        "classes_vi": config.CLASS_NAMES_VI,
        "samples": get_sample_images(),
        "model_trained": bool(predictor is not None and predictor.is_trained),
        "max_upload_mb": config.MAX_UPLOAD_SIZE_MB,
    }
    return templates.TemplateResponse(request=request, name="index.html", context=context)


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint returning model status, model metadata and configuration.
    """
    metadata = dict(predictor.metadata) if predictor is not None else {}

    return {
        "status": "online",
        "model_loaded": predictor is not None,
        "model_trained": bool(predictor is not None and predictor.is_trained),
        "model_checkpoint_exists": os.path.exists(config.MODEL_PATH),
        "model_path": config.MODEL_PATH,
        "device": str(config.DEVICE),
        "target_classes": config.TARGET_CLASSES,
        "loaded_classes": list(predictor.class_names) if predictor is not None else [],
        "classes_vi": config.CLASS_NAMES_VI,
        "model": {
            "arch": metadata.get('arch'),
            "use_pretrained": metadata.get('use_pretrained'),
            "trained_at": metadata.get('trained_at'),
            "best_epoch": metadata.get('epoch'),
            "best_val_acc": metadata.get('val_acc'),
            "num_classes": metadata.get('num_classes'),
            "image_size": metadata.get('image_size'),
            "batch_size": metadata.get('batch_size'),
            "learning_rate": metadata.get('learning_rate'),
        } if metadata else None,
        "samples_available": [s['key'] for s in get_sample_images()],
        "max_upload_mb": config.MAX_UPLOAD_SIZE_MB,
        "allowed_extensions": list(config.ALLOWED_UPLOAD_EXTENSIONS),
    }


@app.get("/api/samples")
async def list_samples():
    """
    Liệt kê ảnh mẫu thật có sẵn (thay cho ảnh vẽ bằng canvas trước đây).

    Trả về danh sách rỗng nếu chưa chạy `python scripts/make_sample_images.py`.
    """
    samples = get_sample_images()
    return {
        "success": True,
        "count": len(samples),
        "samples": samples,
    }


@app.post("/api/predict")
async def predict_file(file: UploadFile = File(...)):
    """
    Predict flower class from an uploaded image file.

    Trả về 400/413 nếu file không hợp lệ, 503 nếu model chưa sẵn sàng.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Mô hình chưa được khởi tạo xong. Vui lòng thử lại sau vài giây.",
        )

    filename = file.filename or 'unnamed'
    extension = os.path.splitext(filename)[1].lower()

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File '{filename}' không phải ảnh (content-type: {file.content_type}).",
        )

    if extension and extension not in config.ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Định dạng '{extension}' không được hỗ trợ. "
                f"Chỉ nhận: {', '.join(config.ALLOWED_UPLOAD_EXTENSIONS)}."
            ),
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="File rỗng, không có dữ liệu ảnh.")

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Ảnh vượt quá giới hạn {config.MAX_UPLOAD_SIZE_MB}MB "
                f"(kích thước thực tế: {len(contents) / (1024 * 1024):.1f}MB)."
            ),
        )

    image = _open_uploaded_image(contents)

    if not predictor.is_trained:
        logger.warning("Dự đoán '%s' bằng model CHƯA huấn luyện.", filename)

    try:
        prediction_result = predictor.predict_image(image)
    except Exception as e:
        logger.exception("Lỗi khi dự đoán ảnh '%s'", filename)
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán ảnh: {e}")

    prediction_result['filename'] = filename
    prediction_result['image_size'] = list(image.size)

    return JSONResponse(content={
        "success": True,
        "data": prediction_result,
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
