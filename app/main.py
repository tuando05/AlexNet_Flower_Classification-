import io
import os
import sys
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Ensure root directory is in python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from src.inference import FlowerPredictor

app = FastAPI(
    title="Phân Loại Ảnh Hoa AlexNet API",
    description="REST API và Web App phân loại 3 loại hoa (Hoa Hồng, Hoa Cúc, Hoa Ly) bằng mô hình Deep Learning AlexNet",
    version="1.0.0"
)

# Mount Static Files and Templates
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize Predictor
predictor = None

@app.on_event("startup")
async def startup_event():
    global predictor
    print("Initializing FlowerPredictor instance...")
    predictor = FlowerPredictor(model_path=config.MODEL_PATH, device=config.DEVICE)
    print("FlowerPredictor initialized successfully.")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Renders the main modern web UI dashboard.
    """
    context = {
        "classes": config.TARGET_CLASSES,
        "classes_vi": config.CLASS_NAMES_VI
    }
    return templates.TemplateResponse(request=request, name="index.html", context=context)


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint returning model status and configuration.
    """
    model_exists = os.path.exists(config.MODEL_PATH)
    return {
        "status": "online",
        "model_loaded": predictor is not None,
        "model_checkpoint_exists": model_exists,
        "model_path": config.MODEL_PATH,
        "target_classes": config.TARGET_CLASSES,
        "device": str(config.DEVICE)
    }


@app.post("/api/predict")
async def predict_file(file: UploadFile = File(...)):
    """
    Predict flower class from an uploaded image file.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not a valid image format.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        if predictor is None:
            raise HTTPException(status_code=500, detail="Model predictor is not initialized.")

        prediction_result = predictor.predict_image(image)
        prediction_result['filename'] = file.filename
        
        return JSONResponse(content={
            "success": True,
            "data": prediction_result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image prediction: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
