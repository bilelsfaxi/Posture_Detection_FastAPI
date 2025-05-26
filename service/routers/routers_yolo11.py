from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import Response
from service.schemas.schemas_yolo11 import DetectionResponse
from service.detectors.detectors_yolo11 import YOLOv11Detector
import os
import tempfile
import numpy as np
from io import BytesIO
from PIL import Image
import cv2

router = APIRouter(prefix="/yolo", tags=["YOLOv11"])

@router.post("/predict", response_model=DetectionResponse)
async def predict(file: UploadFile = File(...)):
    # Valider le type de fichier
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Initialiser le détecteur
        detector = YOLOv11Detector()
        
        # Lire l'image en mémoire
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        # Utiliser un fichier temporaire pour la sortie
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_output:
            temp_output_path = temp_output.name
            # Traiter l'image
            detections = detector.process_image(image_np, temp_output_path)

        # Vérifier si l'image annotée a été créée
        if not os.path.exists(temp_output_path):
            raise HTTPException(status_code=500, detail="Failed to generate annotated image")

        # Lire l'image annotée et la renvoyer
        with open(temp_output_path, "rb") as annotated_file:
            encoded_image = annotated_file.read()

        # Nettoyer le fichier temporaire
        os.remove(temp_output_path)

        return Response(content=encoded_image, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")