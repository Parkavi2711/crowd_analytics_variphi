# Crowd Analytics

A real-time crowd monitoring system that uses YOLOv8 object detection to count people in configurable zones, estimate crowd flow direction using optical flow, and generate threshold-based alerts.

## Deliverables Checklist

| Deliverable | Location |
|---|---|
| Source code (clean folder structure) | `crowd_analytics/src/` |
| README.md | `README.md` (this file) |
| DESIGN.md | `DESIGN.md` |
| ONNX model file | `crowd_analytics/src/yolov8n.onnx` |
| ONNX Runtime benchmark | `benchmark_onnx.py`, `benchmark_results.md`, `benchmark_results_final.md` |
| Training summary | [Colab Notebook](https://colab.research.google.com/drive/13Qmzcx_gg2xaNRGn4CuliEdFfybyHTe_?usp=sharing) |
| Sample output (annotated video) | `crowd_analytics/outputs/final/annotated.avi` |

## Project Structure

```
crowd/
├── README.md                        # This file
├── DESIGN.md                        # Design document
├── requirements.txt                 # Python dependencies
├── coordinates.py                   # Interactive coordinate picker tool
├── benchmark_pt.py                  # PyTorch inference benchmark
├── benchmark_onnx.py                # ONNX Runtime inference benchmark
├── benchmark_results_video2.md      # Benchmark results — video2.mp4
├── benchmark_results_final.md       # Benchmark results — final.mp4
├── crowd_analytics/
│   ├── assets/
│   │   ├── video2.mp4               # Test video 1 (1280×720, landscape)
│   │   └── final.mp4                # Test video 2 (1080×1920, portrait)
│   ├── config/
│   │   ├── zones.yaml               # Zone polygons for video2
│   │   ├── zone_new.yaml            # Zone polygons for final
│   │   ├── thresholds.yaml          # Crowd thresholds for video2
│   │   └── threshold_new.yaml       # Crowd thresholds for final
│   ├── outputs/
│   │   ├── video2/
│   │   │   ├── annotated.avi        # Annotated output for video2
│   │   │   └── alerts.jsonl         # Alerts for video2
│   │   └── final/
│   │       ├── annotated.avi        # Annotated output for final
│   │       └── alerts.jsonl         # Alerts for final
│   └── src/
│       ├── pipeline.py              # Main processing pipeline
│       ├── detector.py              # YOLOv8 person detector
│       ├── zones.py                 # Zone counting with Shapely
│       ├── flow.py                  # Optical flow estimator
│       ├── utils.py                 # Smoothing, direction, drawing helpers
│       ├── yolov8n.pt               # YOLOv8n PyTorch weights
│       └── yolov8n.onnx             # YOLOv8n ONNX weights
└── crowd_env/                       # Python virtual environment
```

## Setup Instructions

### Prerequisites

- Python 3.10+

### Installation

```bash
python -m venv crowd_env
crowd_env\Scripts\activate            # Windows
# source crowd_env/bin/activate       # Linux/macOS

pip install -r requirements.txt
```

## How to Run on the Test Videos

### Run on `final.mp4` (default)

```bash
cd crowd_analytics/src
python pipeline.py
```

This uses the default configs (`zone_new.yaml` + `threshold_new.yaml`) and writes output to `outputs/final/`.

### Run on `video2.mp4`

Update `VIDEO_PATH` in `pipeline.py` to `video2.mp4`, then override configs:

```bash
set ZONE_CONFIG=config/zones.yaml
set THRESHOLD_CONFIG=config/thresholds.yaml
cd crowd_analytics/src
python pipeline.py
```

Output is written to `outputs/video2/`.

### Output Files

Each video produces two output files in its subfolder under `outputs/`:

- **`annotated.avi`** — Video with bounding boxes, zone overlays, per-zone counts, and flow direction.
- **`alerts.jsonl`** — One JSON object per line for each frame where a zone exceeds its threshold:

```json
{"timestamp": "11:19:00", "zone": "Left_Concourse_Area", "count": 20, "threshold": 18}
```

## How to Configure Zones and Thresholds

### Step 1: Find Zone Coordinates

Use the interactive coordinate picker to click on the video frame and capture polygon vertices:

```bash
python coordinates.py
```

Click on the frame to print `[x, y]` coordinates, press `q` to quit.

### Step 2: Define Zones

Create or edit a zone YAML file in `crowd_analytics/config/`. Each zone is a named polygon:

```yaml
zones:
  - name: Ticket_Gate_Area
    polygon:
      - [192, 461]
      - [856, 304]
      - [812, 675]
      - [186, 811]
```

### Step 3: Set Thresholds

Create or edit a threshold YAML file. Map each zone name to its maximum allowed person count:

```yaml
Ticket_Gate_Area: 4
Left_Concourse_Area: 18
Right_Entry_Exit_Gates: 6
```

### Step 4: Point Pipeline to Your Configs

Set environment variables before running:

```bash
set ZONE_CONFIG=config/your_zones.yaml
set THRESHOLD_CONFIG=config/your_thresholds.yaml
python pipeline.py
```

### Video-Specific Configurations

| Video | Resolution | Zone Config | Threshold Config |
|---|---|---|---|
| `video2.mp4` | 1280×720 (landscape) | `zones.yaml` | `thresholds.yaml` |
| `final.mp4` | 1080×1920 (portrait) | `zone_new.yaml` (default) | `threshold_new.yaml` (default) |

## ONNX Runtime Benchmark (CPU-Only)

### `final.mp4` (1080×1920, 893 frames)

| Metric | PyTorch (.pt) | ONNX Runtime (.onnx) |
|---|---|---|
| Avg per frame | 51.61 ms | 57.47 ms |
| FPS | 19.38 | 17.40 |

### `video2.mp4` (1280×720, 631 frames)

| Metric | PyTorch (.pt) | ONNX Runtime (.onnx) |
|---|---|---|
| Avg per frame | 42.28 ms | 45.41 ms |
| FPS | 23.65 | 22.02 |

Full results: `benchmark_results_video2.md` and `benchmark_results_final.md`.

## Sample Output

Pre-generated annotated videos are available at:

- `crowd_analytics/outputs/final/annotated.avi` — final.mp4 with detections, zone overlays, counts, and flow.
- `crowd_analytics/outputs/video2/annotated.avi` — video2.mp4 with detections, zone overlays, counts, and flow.

## Training Summary

We attempted to fine-tune YOLOv8n on the [CrowdHuman](https://www.crowdhuman.org/) dataset to improve person detection in dense crowd scenarios.

**Colab Notebook:** [View on Google Colab](https://colab.research.google.com/drive/13Qmzcx_gg2xaNRGn4CuliEdFfybyHTe_?usp=sharing)

### Dataset Preparation

- **Dataset:** CrowdHuman (crowd-only subset)
- **Split:** 6,000 train images, 749 validation images
- **Class:** Single class — `person` (class 0)
- **Label format:** YOLO format (`class cx cy w h`, normalized), converted from CrowdHuman annotations
- **Config:** `crowd_only.yaml` with train/val/test paths and `nc: 1`

### Training Configuration

| Parameter | Value |
|---|---|
| Base model | `yolov8n.pt` (pretrained on COCO) |
| Epochs | 50 (target) |
| Image size | 640 |
| Batch size | 16 |
| Optimizer | AdamW |
| Device | Tesla T4 GPU (Google Colab) |
| Workers | 4 |

### Results & GPU Limitation

Training progressed through approximately **8 out of 50 epochs** before being interrupted due to Google Colab's GPU runtime limits (session timeout / GPU quota exhaustion on the free tier).

**Metrics at epoch 8:**

| Metric | Value |
|---|---|
| Box loss | 2.691 |
| Class loss | 3.606 |
| DFL loss | 2.027 |
| mAP50 | 0.00103 |
| mAP50-95 | 0.000356 |

The low mAP values are expected at this early stage — YOLOv8 typically requires 30+ epochs to converge, and CrowdHuman's dense, heavily-occluded scenes are particularly challenging. With a full 50-epoch run or a Colab Pro subscription, significantly better metrics would be expected.

### Current Approach

Due to the incomplete fine-tuning, the production pipeline uses the **pretrained YOLOv8n model** (trained on COCO, which includes the person class). This model performs well for moderate crowd densities and is sufficient for the zone-counting use case demonstrated in this project.

## Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 inference |
| `opencv-python` | Video I/O, drawing, optical flow |
| `pyyaml` | YAML config parsing |
| `shapely` | Point-in-polygon zone checks |
| `numpy` | Array operations |
| `onnxruntime` | ONNX model inference & benchmarking |