# YOLOv8n Benchmark Results — final.mp4

**Date:** April 22, 2026
**Hardware:** CPU only
**Model:** YOLOv8n (nano)
**Video:** final.mp4, 893 frames, 1080×1920 (portrait)
**Confidence Threshold:** 0.15
**Detection Class:** Person only (class 0)

## Results

| Metric | PyTorch (.pt) | ONNX Runtime (.onnx) |
|---|---|---|
| Frames processed | 893 | 893 |
| Total time | 46.08s | 51.32s |
| Avg per frame | 51.61 ms | 57.47 ms |
| FPS | 19.38 | 17.40 |

## Comparison with video2.mp4 (1280×720)

| Metric | video2.mp4 (.pt) | video2.mp4 (.onnx) | final.mp4 (.pt) | final.mp4 (.onnx) |
|---|---|---|---|---|
| Resolution | 1280×720 | 1280×720 | 1080×1920 | 1080×1920 |
| Frames | 631 | 631 | 893 | 893 |
| Total time | 26.68s | 28.66s | 46.08s | 51.32s |
| Avg per frame | 42.28 ms | 45.41 ms | 51.61 ms | 57.47 ms |
| FPS | 23.65 | 22.02 | 19.38 | 17.40 |

## Conclusion

- **PyTorch (.pt) is consistently faster** than ONNX Runtime on CPU across both videos — ~11% faster on final.mp4 and ~7% faster on video2.mp4.
- **Higher resolution increases latency** — the portrait 1080×1920 video (final.mp4) is ~22% slower per frame than the 1280×720 video (video2.mp4) using PyTorch, and ~27% slower using ONNX.
- Both formats remain under 60 ms per frame, making them viable for offline processing. However, only video2.mp4 achieves real-time speeds (>20 FPS).
- For real-time processing of higher-resolution videos, GPU acceleration or frame resizing before inference would be recommended.
