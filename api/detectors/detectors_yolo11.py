from ultralytics import YOLO
import os
import cv2
import numpy as np
from typing import List, Dict

class YOLOv11Detector:
    def __init__(self, model_path: str = os.path.join(os.path.dirname(__file__), "..", "models", "final_model_yolo11.pt")):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Le modèle YOLOv11 n’a pas été trouvé à l’emplacement : {model_path}\n"
                f"Veuillez vous assurer que le fichier .pt est présent dans le dossier du projet."
            )

        try:
            self.model = YOLO(model_path, task="detect")
            self.classes = self.model.names
            print(f"✅ Modèle chargé avec succès depuis {model_path}")
        except Exception as e:
            raise RuntimeError(f"❌ Échec du chargement du modèle : {str(e)}")

    def process_image(self, image_np: np.ndarray, output_path: str = None) -> List[Dict]:
        results = self.model(image_np, conf=0.5)
        detections = []
        annotated_image = image_np.copy()

        for r in results:
            for box in r.boxes:
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

        # Utiliser un codec compatible avec les navigateurs (H.264)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Alternative : 'H264' ou 'X264'
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        all_detections = []

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Fin de la vidéo après {frame_count} frames")
                break

            detections = self.process_image(frame)
            all_detections.extend(detections)
            annotated_frame = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                text_y = max(y1 - 10, 20)
                cv2.putText(annotated_frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            out.write(annotated_frame)
            frame_count += 1
            print(f"Frame {frame_count} annotée et écrite")

        cap.release()
        out.release()
        print(f"Vidéo annotée sauvegardée à {output_path}")
        return all_detections