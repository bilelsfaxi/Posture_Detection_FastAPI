from fastapi import FastAPI
from uvicorn import Server, Config
import os

from service.routers import routers_yolo11

app = FastAPI(title="YOLOv11 Dog Posture Detection API")

# Inclure les routes définies dans routers/yolo.py
app.include_router(routers_yolo11.router)

@app.get("/")
async def root():
    return {"message": "YOLOv11 Dog Posture Detection API. Use POST /yolo/predict to upload an image."}
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 80))
    server = Server(Config(app, host="0.0.0.0", port=port, lifespan="on"))
    server.run()