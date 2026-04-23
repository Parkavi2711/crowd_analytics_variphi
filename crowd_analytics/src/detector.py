from ultralytics import YOLO

class PersonDetector:
    def __init__(self, conf=0.15):
        self.model = YOLO("yolov8n.pt")
        self.conf = conf

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            classes=[0],      # ✅ PERSON ONLY
            verbose=False
        )

        if not results or results[0].boxes is None:
            return []

        return results[0].boxes.xyxy.cpu().numpy()