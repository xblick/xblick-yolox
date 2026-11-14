# Changelog

## 0.1.2 (2026-11-14)

- The weights moved out of the checkout into the node's cache directory, `~/.cache/xblick/yolox/`. `XBLICK_HOME` and
  `XDG_CACHE_HOME` move it, and `--weights-dir DIR` names another directory. Weights at the former place, the
  checkout's `weights/`, are not used. The node prints the command that moves them. In the checkout it is
  `mkdir -p ~/.cache/xblick && mv weights ~/.cache/xblick/yolox`.
- `--weights-dir DIR` is a new start option and a read-only parameter.
- The message about a missing file names the full download command (`xblick-yolox --download-only --model tiny`).
- The node needs `xblick>=0.5.5,<0.6`.
- `weights.path_of(model, store)` gives the model's file in a store, the node's cache directory by default.
- The README's quick start installs nothing of the core's `cam` extra. It tells you to install it separately,
  `pip install "xblick[cam]"`, for `xblick cam`.

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
