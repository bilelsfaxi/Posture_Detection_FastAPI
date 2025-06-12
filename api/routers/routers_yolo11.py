from fastapi import APIRouter, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse
from api.schemas.schemas_yolo11 import DetectionResponse
from websockets.exceptions import ConnectionClosed
import os
import tempfile
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import asyncio

router = APIRouter(prefix="/yolo", tags=["YOLOv11"])

# Détecteur injecté par main.py
detector = None

# Source vidéo globale
video_source = None
streaming_active = False  # Contrôle l'état du streaming

@router.post("/predict", response_model=DetectionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        if detector is None:
            raise RuntimeError("Le modèle YOLOv11 n'a pas été initialisé")

        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_output:
            temp_output_path = temp_output.name
            detections = detector.process_image(image_np, temp_output_path)

        if not os.path.exists(temp_output_path):
            raise HTTPException(status_code=500, detail="Échec de la génération de l'image annotée")

        with open(temp_output_path, "rb") as annotated_file:
            encoded_image = annotated_file.read()

        os.remove(temp_output_path)

        return Response(content=encoded_image, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement de l'image : {str(e)}")

@router.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    global video_source
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    try:
        if detector is None:
            raise RuntimeError("Le modèle YOLOv11 n'a pas été initialisé")

        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(contents)
            temp_input_path = temp_input.name

        video_source = cv2.VideoCapture(temp_input_path)
        if not video_source.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo à {temp_input_path}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
            temp_output_path = temp_output.name
            detector.process_video(temp_input_path, temp_output_path)

        if video_source is not None:
            video_source.release()
            video_source = None

        if not os.path.exists(temp_output_path):
            raise HTTPException(status_code=500, detail="Fichier vidéo annoté non généré")

        with open(temp_output_path, "rb") as output_file:
            encoded_video = output_file.read()

        os.remove(temp_input_path)
        os.remove(temp_output_path)

        return Response(content=encoded_video, media_type="video/mp4")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement de la vidéo : {str(e)}")
        if video_source is not None:
            video_source.release()
            video_source = None

@router.post("/start-stream")
async def start_stream():
    global video_source, streaming_active
    if not streaming_active:
        video_source = cv2.VideoCapture(0)
        if not video_source.isOpened():
            return JSONResponse(status_code=500, content={"message": "Impossible d'ouvrir la webcam"})
        streaming_active = True
        return JSONResponse(content={"message": "Streaming activé"})
    return JSONResponse(content={"message": "Streaming déjà actif"})

@router.post("/stop-stream")
async def stop_stream():
    global video_source, streaming_active
    if streaming_active:
        if video_source is not None:
            video_source.release()
            video_source = None
        streaming_active = False
        return JSONResponse(content={"message": "Streaming arrêté"})
    return JSONResponse(content={"message": "Streaming déjà arrêté"})

@router.websocket("/ws")
async def stream_video(websocket: WebSocket):
    """Diffuse les frames annotés via WebSocket uniquement si activé."""
    global video_source, streaming_active
    await websocket.accept()
    try:
        while True:
            if not streaming_active or video_source is None or not video_source.isOpened():
                await websocket.send_text("Streaming désactivé")
                await asyncio.sleep(1)
                continue
            success, frame = video_source.read()
            if not success:
                print("Fin de la vidéo ou erreur de lecture")
                streaming_active = False
                await websocket.send_text("Streaming arrêté")
                break
            frame = cv2.resize(frame, (640, 360))
            detections = detector.process_image(frame)
            annotated_frame = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                text_y = max(y1 - 10, 20)
                cv2.putText(annotated_frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                print("Échec de l'encodage JPEG")
                continue
            await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.03)
    except (WebSocketDisconnect, ConnectionClosed):
        print("Client disconnected")
    finally:
        if video_source is not None:
            video_source.release()
            video_source = None
            streaming_active = False
            print("Vidéo libérée")