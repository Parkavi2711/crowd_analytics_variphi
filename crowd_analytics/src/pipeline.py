import cv2
import os
import json
import time
import yaml

from detector import PersonDetector
from zones import ZoneCounter
from flow import FlowEstimator
from utils import smooth, flow_to_direction, draw_zones

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_PATH = os.path.join(BASE_DIR, "assets", "final.mp4")

ZONE_CONFIG = os.environ.get(
    "ZONE_CONFIG",
    os.path.join(BASE_DIR, "config", "zone_new.yaml")
)

THRESHOLD_CONFIG = os.environ.get(
    "THRESHOLD_CONFIG",
    os.path.join(BASE_DIR, "config", "threshold_new.yaml")
)
OUTPUT_VIDEO = os.path.join(BASE_DIR, "outputs", "annotated.avi")
ALERT_LOG = os.path.join(BASE_DIR, "outputs", "alerts.jsonl")

with open(THRESHOLD_CONFIG, "r") as f:
    THRESHOLDS = yaml.safe_load(f)

detector = PersonDetector(conf=0.15)
zone_counter = ZoneCounter(ZONE_CONFIG)
flow_estimator = FlowEstimator()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"❌ Cannot open video file: {VIDEO_PATH}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25

print(f"Video size: {w}x{h}, FPS: {fps}")

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (w, h)
)
assert writer.isOpened(), "❌ VideoWriter failed"

prev_counts = {}

with open(ALERT_LOG, "w") as alert_file:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detector.detect(frame)
        print("Detected people:", len(boxes))

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        counts = zone_counter.count(boxes)
        print("Zone counts:", counts)

        smooth_counts = {}
        for zone, count in counts.items():
            smooth_counts[zone] = smooth(
                prev_counts.get(zone),
                count
            )
        prev_counts = smooth_counts

        flow = flow_estimator.estimate(frame)
        direction = flow_to_direction(flow)

        draw_zones(frame, zone_counter.zones)

        y = 40
        for zone, count in smooth_counts.items():
            threshold = THRESHOLDS.get(zone, 9999)
            color = (0, 255, 0)

            if count > threshold:
                color = (0, 0, 255)
                alert = {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "zone": zone,
                    "count": int(count),
                    "threshold": threshold
                }
                alert_file.write(json.dumps(alert) + "\n")

            cv2.putText(
                frame,
                f"{zone}: {int(count)}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            y += 30

        cv2.putText(
            frame,
            f"Flow: {direction}",
            (20, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        writer.write(frame)

cap.release()
writer.release()
cv2.destroyAllWindows()
print("✅ Processing complete")