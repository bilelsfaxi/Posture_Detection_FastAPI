from ultralytics import YOLO
import os
import gdown
import tempfile
import cv2
import numpy as np
from typing import List, Dict

class YOLOv11Detector:
    def __init__(self, model_url: str = "https://drive.google.com/uc?id=17uxGv6mOcy8kYgwutfQFm-HTaJNmDltd"):
        temp_dir = tempfile.gettempdir()
        self.model_path = os.path.join(temp_dir, "final_model_yolo11.pt")
        if not os.path.exists(self.model_path):
            print(f"Téléchargement du modèle depuis {model_url} vers {self.model_path}...")
            try:
                gdown.download(model_url, self.model_path, quiet=False)
                print(f"Modèle téléchargé avec succès.")
            except Exception as e:
                raise RuntimeError(f"Échec du téléchargement du modèle : {str(e)}")

        try:
            self.model = YOLO(self.model_path, task="detect")
            self.classes = self.model.names
            print(f"Modèle chargé avec succès à {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Échec du chargement du modèle : {str(e)}")

    def process_image(self, image_np: np.ndarray, output_path: str = None) -> List[Dict]:
        results = self.model(image_np, conf=0.5)
        detections = []
        annotated_image = image_np.copy()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.classes[class_id]

                    detections.append({
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })

                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {confidence:.2f}"
                    text_y = max(y1 - 10, 20)
                    cv2.putText(annotated_image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                except Exception as e:
                    print(f"Erreur lors du traitement de la boîte : {str(e)}")
                    continue

        if output_path:
            cv2.imwrite(output_path, annotated_image)

        return detections

    def process_video(self, video_path: str, output_path: str) -> List[Dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Impossible d'ouvrir la vidéo")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        all_detections = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.process_image(frame)
            all_detections.extend(detections)

            out.write(frame)

        cap.release()
        out.release()
        return all_detections
