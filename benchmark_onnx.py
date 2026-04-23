import time
import cv2
import numpy as np
import onnxruntime as ort

VIDEO = "crowd_analytics/assets/final.mp4"
MODEL = "crowd_analytics/src/yolov8n.onnx"
CONF = 0.15
IMG_SIZE = 640

session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

cap = cv2.VideoCapture(VIDEO)

frame_count = 0
start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    inp = preprocess(frame)
    session.run(None, {input_name: inp})
    frame_count += 1

end = time.time()
elapsed = end - start
fps = frame_count / elapsed
avg_ms = (elapsed / frame_count) * 1000

print("\n=== YOLO ONNX Benchmark (CPU) ===")
print(f"Frames processed : {frame_count}")
print(f"Total time       : {elapsed:.2f}s")
print(f"Avg per frame    : {avg_ms:.2f} ms")
print(f"FPS              : {fps:.2f}")

cap.release()
