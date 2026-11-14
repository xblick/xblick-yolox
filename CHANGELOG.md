# Changelog

## 0.1.0 (2026-11-14)

The first version: YOLOX (nano, tiny, s) in ONNX Runtime on the CPU, detections as `vision_msgs/Detection2DArray` on
`<camera>/yolox/detections_2d`, the optional overlay `<camera>/yolox/overlay` (`--overlay`), diagnostics, options as
parameters (`score`, `nms`, `classes` live), the model downloaded at the first start (`--download-only`,
`--no-download`).
