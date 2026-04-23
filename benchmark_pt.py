import time
import cv2
from ultralytics import YOLO

VIDEO = "crowd_analytics/assets/final.mp4"
CONF = 0.15

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO)

frame_count = 0
start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    model.predict(frame, conf=CONF, classes=[0], verbose=False)
    frame_count += 1

end = time.time()
elapsed = end - start
fps = frame_count / elapsed
avg_ms = (elapsed / frame_count) * 1000

print("\n=== YOLO .pt Benchmark (CPU) ===")
print(f"Frames processed : {frame_count}")
print(f"Total time       : {elapsed:.2f}s")
print(f"Avg per frame    : {avg_ms:.2f} ms")
print(f"FPS              : {fps:.2f}")

cap.release()
