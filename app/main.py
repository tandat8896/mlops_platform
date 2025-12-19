import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import logging
import torch
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _download_model_from_s3(s3_bucket: str, s3_key: str, local_path: str):
    import boto3
    s3 = boto3.client('s3')
    s3.download_file(s3_bucket, s3_key, local_path)
    logger.info(f"Model downloaded from s3://{s3_bucket}/{s3_key} to {local_path}")


ml_models = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading YOLO model...")

    # Add safe globals for PyTorch 2.6+ compatibility
    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except Exception as e:
        logger.warning(f"Could not add safe globals: {e}")
    
    # Load the ML model
    # ml_models["yolo11n"] = YOLO('yolo11n.pt')
    local_model_path = "yolo11n.pt"
    _download_model_from_s3(s3_bucket=os.getenv("S3_BUCKET"),
                            s3_key="yolo11n-detection/best.pt",
                            local_path=local_model_path)
    ml_models["yolo11n"] = YOLO(local_model_path)
    logger.info("Model loaded successfully!")
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(
    title="YOLO Object Detection API",
    description="FastAPI service for object detection using YOLO model",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "YOLO Object Detection API",
        "status": "running",
        "model": "YOLO11n"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": "yolo11n" in ml_models
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict objects in uploaded image
    
    Args:
        file: Image file (jpg, png, etc.)
    
    Returns:
        JSON with detected objects and their bounding boxes
    """
    if "yolo11n" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model = ml_models["yolo11n"]
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Run inference
        logger.info(f"Running inference on image: {file.filename}")
        results = model(img_array)
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "class": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3])
                    }
                }
                detections.append(detection)
        
        logger.info(f"Found {len(detections)} objects")
        
        return JSONResponse(content={
            "filename": file.filename,
            "detections": detections,
            "count": len(detections)
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model-info")
async def model_info():
    """Get model information"""
    if "yolo11n" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model = ml_models["yolo11n"]
    return {
        "model_name": "YOLO11n",
        "classes": list(model.names.values()),
        "num_classes": len(model.names)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
