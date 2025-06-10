from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import Response
from api.schemas.schemas_yolo11 import DetectionResponse
import os
import tempfile
import numpy as np
from io import BytesIO
from PIL import Image

router = APIRouter(prefix="/yolo", tags=["YOLOv11"])

# Ce champ sera injecté par main.py
detector = None

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
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    try:
        if detector is None:
            raise RuntimeError("Le modèle YOLOv11 n'a pas été initialisé")

        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(contents)
            temp_input_path = temp_input.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
            temp_output_path = temp_output.name
            detector.process_video(temp_input_path, temp_output_path)

        with open(temp_output_path, "rb") as output_file:
            encoded_video = output_file.read()

        os.remove(temp_input_path)
        os.remove(temp_output_path)

        return Response(content=encoded_video, media_type="video/mp4")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement de la vidéo : {str(e)}")
