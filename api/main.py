from fastapi import FastAPI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .routers import routers_yolo11

app = FastAPI(title="YOLOv11 Dog Posture Detection API")
app.include_router(routers_yolo11.router)

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup complete. Server is running.")

@app.get("/")
async def root():
    return {"message": "YOLOv11 Dog Posture Detection API. Use POST /yolo/predict to upload an image."}