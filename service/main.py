from fastapi import FastAPI
from service.routers import routers_yolo11

app = FastAPI(title="YOLOv11 Dog Posture Detection API")

# Inclure les routes définies dans routers/yolo.py
app.include_router(routers_yolo11.router)

@app.get("/")
async def root():
    return {"message": "YOLOv11 Dog Posture Detection API. Use POST /yolo/predict to upload an image."}