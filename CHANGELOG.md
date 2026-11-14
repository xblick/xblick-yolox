# Changelog

## 0.1.1 (2026-11-14)

- The core `xblick` 0.5.3 is the minimum. Its `xblick.weights` downloads the model (the node's own copy of that code
  is gone) and its `xblick.overlay` draws and publishes the overlay (the node's own thread is gone).
- The overlay now carries the colour stream's camera_info.
- `--device cuda` loads the CUDA and cuDNN libraries of NVIDIA's wheels and runs the model once at the start. A GPU
  that cannot compute ends the start with one line. Tested with `onnxruntime-gpu[cuda,cudnn]` 1.32.0 in place of
  `onnxruntime`.
- The README starts the node with `xblick router` and shows it with `xblick view`.

## 0.1.0 (2026-11-14)

The first version: YOLOX (nano, tiny, s) in ONNX Runtime on the CPU, detections as `vision_msgs/Detection2DArray` on
`<camera>/yolox/detections_2d`, the optional overlay `<camera>/yolox/overlay` (`--overlay`), diagnostics, options as
parameters (`score`, `nms`, `classes` live), the model downloaded at the first start (`--download-only`,
`--no-download`).
