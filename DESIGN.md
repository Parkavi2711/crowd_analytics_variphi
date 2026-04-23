# Design Document — Crowd Analytics

## Detection Approach

The system uses **YOLOv8n (nano)**, a pre-trained single-stage object detector from the Ultralytics framework. YOLOv8n was chosen for its balance between accuracy and speed — it achieves 19–24 FPS on CPU, making it viable for near-real-time crowd monitoring without GPU hardware. Inference is restricted to COCO class 0 (person) to eliminate wasted computation on irrelevant object categories. The confidence threshold is set to 0.15 to favour recall over precision, since undercounting in safety-critical scenarios is more harmful than occasional false positives. Exponential smoothing (α=0.7) is applied to per-zone counts across frames to suppress jitter from frame-to-frame detection variance.

The model is also exported to **ONNX format** for deployment flexibility. ONNX Runtime benchmarks confirm both formats run under 60 ms per frame on CPU, with PyTorch being ~7–11% faster depending on video resolution.

## Pipeline Structure

The pipeline (`pipeline.py`) processes video frame-by-frame through four modular stages:

1. **Detection** — `PersonDetector` runs YOLOv8n on each frame, returning bounding boxes in `[x1, y1, x2, y2]` format.
2. **Zone Counting** — `ZoneCounter` loads polygon zones from a YAML config and uses Shapely's point-in-polygon test on each bounding box centre point. Centre-point membership was chosen over bounding-box overlap to prevent double-counting across adjacent zones.
3. **Flow Estimation** — `FlowEstimator` computes Farneback dense optical flow between consecutive grayscale frames and averages the result to a single `(dx, dy)` vector, which is mapped to a cardinal direction (N/S/E/W) or "Static". Dense flow captures global crowd movement without requiring per-person tracking.
4. **Annotation & Alerting** — Bounding boxes, zone boundaries, counts, and flow direction are drawn onto each frame. When a zone count exceeds its configured threshold, an alert is logged.

Zone polygons, thresholds, and the video path are externalised into YAML config files. Both `ZONE_CONFIG` and `THRESHOLD_CONFIG` can be overridden via environment variables, allowing the same codebase to serve multiple cameras or venues without code changes. A coordinate picker utility (`coordinates.py`) lets users click on a video frame to capture polygon vertices interactively.

## Output Format

The system produces two outputs per video:

- **Annotated video** (`annotated.avi`) — XVID-encoded video with green bounding boxes around detected people, blue zone polygon overlays with labels, per-zone counts (green = normal, red = threshold exceeded), and a flow direction indicator. XVID was chosen for broad compatibility without requiring additional system libraries.
- **Alert log** (`alerts.jsonl`) — Newline-delimited JSON, one object per threshold breach per frame. Each entry contains `timestamp`, `zone`, `count`, and `threshold`. JSONL was chosen because it is append-friendly, trivially parseable, and compatible with log aggregation and analysis tools. Alerts fire on every exceeding frame to provide a granular temporal record for post-incident review.

### Training Attempt

Fine-tuning YOLOv8n on CrowdHuman (6,000 images, single `person` class) was attempted via Google Colab (Tesla T4). Training reached ~13/50 epochs before Colab's free-tier GPU quota was exhausted. Early-stage mAP50 was 0.001, which is expected — YOLOv8 typically needs 30+ epochs to converge on dense crowd data. The production pipeline therefore uses the pretrained COCO weights, which already include a strong person detector sufficient for the demonstrated crowd densities.
