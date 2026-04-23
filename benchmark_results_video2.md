# YOLOv8n Benchmark Results

**Date:** April 22, 2026
**Hardware:** CPU only
**Model:** YOLOv8n (nano)
**Video:** 631 frames, 1280×720
**Confidence Threshold:** 0.15
**Detection Class:** Person only (class 0)

## Results

| Metric | PyTorch (.pt) | ONNX Runtime (.onnx) |
|---|---|---|
| Frames processed | 631 | 631 |
| Total time | 26.68s | 28.66s |
| Avg per frame | 42.28 ms | 45.41 ms |
| FPS | 23.65 | 22.02 |

## Conclusion

- **PyTorch (.pt) is faster on CPU** — it achieved 23.65 FPS compared to ONNX Runtime's 22.02 FPS, a ~7% speed advantage.
- Both formats run comfortably above 20 FPS on CPU, making either suitable for near-real-time crowd analytics on this video resolution.
- The performance gap is small enough that the choice between formats should be driven by deployment needs rather than raw speed. ONNX is preferable when deploying without a PyTorch dependency, while .pt is simpler during development.
- For higher throughput requirements, GPU acceleration or model quantization (INT8) would provide significantly larger gains than switching inference formats alone.
- On CPU at 1280×720 resolution, PyTorch YOLO slightly outperformed ONNX Runtime in this specific setup, but the difference is marginal and both achieve real‑time performance. This indicates that model format alone is not the primary determinant of throughput; deployment context and optimization strategy matter more. ONNX remains preferable for production integration despite similar raw performance.
