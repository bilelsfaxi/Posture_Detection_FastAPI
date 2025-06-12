from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from websockets.exceptions import ConnectionClosed
import os
import logging
import cv2
import numpy as np
import asyncio
from api.detectors import detectors_yolo11
from api.routers import routers_yolo11

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLOv11 Dog Posture Detection API")
templates = Jinja2Templates(directory="templates")

# Chemin du modèle
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "final_model_yolo11.pt")
detector = detectors_yolo11.YOLOv11Detector(model_path=MODEL_PATH)

# Injection dans le routeur
routers_yolo11.detector = detector

# Source vidéo globale (webcam ou fichier)
video_source = None

app.include_router(routers_yolo11.router)

@app.on_event("startup")
async def startup_event():
    logger.info("Application démarrée avec succès.")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Affiche l'interface web."""
    return templates.TemplateResponse("index.html", {"request": request})
