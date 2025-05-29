from fastapi import FastAPI
import logging
from api.detectors import detectors_yolo11 # Importez le module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from api.routers import routers_yolo11

app = FastAPI(title="YOLOv11 Dog Posture Detection API")
app.include_router(routers_yolo11.router)

# Charger le modèle au démarrage
detector = detectors_yolo11.YOLOv11Detector(model_url="https://drive.google.com/uc?id=17uxGv6mOcy8kYgwutfQFm-HTaJNmDltd")

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup complete. Server is running.")

@app.get("/")
async def root():
    return {"message": "YOLOv11 Dog Posture Detection API. Use POST /yolo/predict to upload an image."}