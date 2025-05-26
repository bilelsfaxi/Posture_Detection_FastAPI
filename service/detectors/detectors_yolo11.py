from ultralytics import YOLO
import os
import cv2
import numpy as np
from typing import List, Dict

class YOLOv11Detector:
    def __init__(self, model_path: str = "final_model_yolo11.onnx"):
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", model_path)
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model file not found at {model_path}. "
                f"Please place the file 'final_model_yolo11.onnx' in the 'service/models/' directory."
            )
        # Charger le modèle avec Ultralytics YOLO
        self.model = YOLO(model_path)
        # Vérifier les classes du modèle
        self.classes = self.model.names
        print(f"Classes du modèle : {self.classes}")

    def process_image(self, image_np: np.ndarray, output_path: str = None) -> List[Dict]:
        # Prétraitement et prédiction avec Ultralytics YOLO
        results = self.model(image_np, conf=0.5)  # Seuil de confiance à 0.5 (ajustable)

        # Liste pour stocker les détections
        detections = []
        annotated_image = image_np.copy()

        # Traiter les résultats
        for r in results:
            boxes = r.boxes
            for box in boxes:
                try:
                    # Extraire les coordonnées, confiance et classe
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordonnées bbox [x1, y1, x2, y2]
                    confidence = float(box.conf[0])  # Confiance
                    class_id = int(box.cls[0])  # Classe prédite
                    class_name = self.classes[class_id]  # Nom de la classe

                    print(f"Détection : {class_name}, Confiance : {confidence}, Bbox : [x1={x1}, y1={y1}, x2={x2}, y2={y2}]")

                    # Ajouter la détection à la liste
                    detections.append({
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })

                    # Dessiner la bounding box et l'étiquette sur l'image
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {confidence:.2f}"
                    text_y = max(y1 - 10, 20)
                    cv2.putText(annotated_image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                except Exception as e:
                    print(f"Erreur lors du traitement de la boîte : {str(e)}")
                    continue

        # Sauvegarder l'image annotée si un chemin de sortie est fourni
        if output_path:
            cv2.imwrite(output_path, annotated_image)

        print(f"Nombre total de détections : {len(detections)}")
        return detections