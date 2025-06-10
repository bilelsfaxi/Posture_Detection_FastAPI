# main.py
from fastapi import FastAPI
import logging
from api.detectors import detectors_yolo11
from api.routers import routers_yolo11

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLOv11 Dog Posture Detection API")

# Modèle local
MODEL_PATH = "api/models/final_model_yolo11.pt"
detector = detectors_yolo11.YOLOv11Detector(model_path=MODEL_PATH)

# Injection dans le routeur
routers_yolo11.detector = detector

app.include_router(routers_yolo11.router)

@app.on_event("startup")
async def startup_event():
    logger.info("Application démarrée avec succès.")

@app.get("/")
async def root():
    return {
        "message": "YOLOv11 Dog Posture Detection API. "
                   "POST /yolo/predict pour image ou /yolo/predict-video pour vidéo."
    }
