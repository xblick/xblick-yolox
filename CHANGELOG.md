# Changelog

## Branch ros2 (not released)

The node with the core's option `--ros` (ROS 2 on `rmw_zenoh`).

### Changed

- The core comes from its unreleased branch `ros2` (`xblick @ git+https://github.com/xblick/xblick@ros2`), not from
  PyPI. `uv sync` builds its C++ part from source.

### Added

- `--ros`, `--ros-domain`, `--ros-node` and `--ros-params-writable` from the core. With `--ros` the detections, the
  overlay and the diagnostics also go to ROS 2, and the parameters are a ROS node. README: section "ROS 2".
- `tests/test_ros.py`: with `--ros` a ROS subscriber gets the detections and the overlay, with ROS's attachment and
  tokens; without it nothing of ROS is on the wire.

Compatible with earlier clients: nothing on our keys changed.

## 0.1.5 (2026-11-15)

The MPS check of the other nodes, and how to cite YOLOX.

### Changed

- README: how to cite YOLOX.

### Fixed

- With `--device cuda` and `CUDA_MPS_PIPE_DIRECTORY` set, the node stops at the start when no MPS daemon runs.

Compatible with earlier clients: nothing on the wire changed.

## 0.1.4 (2026-11-15)

- The GPU runtime for `--device cuda` is shared by the ONNX Runtime nodes: `onnxruntime-gpu` 1.32.0 with NVIDIA's
  CUDA 13 and cuDNN 9 libraries in `~/.cache/xblick/onnxruntime/gpu-1.32.0-cp<Python>/` (`xblick.paths.cache_dir`,
  so `XBLICK_HOME` and `XDG_CACHE_HOME` move it). Nodes that pin the same version use one copy.
- `tools/setup.sh` installs it when the PC has an NVIDIA GPU (`nvidia-smi` lists one, driver 580 or newer), and says
  before the 2.4 GB download what it found and installs where. `--gpu` installs it always, `--no-gpu` never. A second
  call reuses it. The install takes a lock and goes to `<dir>.part`, so an interrupted one never looks complete.
- `--device cuda` prints "preparing the GPU" at once, then loads CUDA and computes once in a thread of its own. Ctrl-C
  and SIGTERM end the node during that time.
- On the GPU every part of the model runs there (ONNX Runtime's CPU fallback is off). Missing runtime, a GPU that
  cannot compute or a Python without a wheel end the start with one line; it names the step.
- The runtime of 0.1.3 at `~/.cache/xblick/yolox/onnxruntime-gpu/` is not used. The node and `tools/setup.sh` print
  the `mv` to the shared place.
- `gpu.py` is now `runtime.py` (the same file in every ONNX Runtime node).
- Python 3.10: `uv.lock` takes onnxruntime 1.23.2, the last version with wheels for 3.10.

## 0.1.3 (2026-11-15)

- `--model` offers all seven models of YOLOX's release 0.1.1rc0: `nano`, `tiny`, `s`, `m`, `l`, `x` and `darknet53`
  (the release's `yolox_darknet.onnx`). Each file is pinned by URL, exact size and SHA-256. The default stays `tiny`.
- The node reads a model's input size from its ONNX file and checks that the output fits it.
- `--device cuda` uses a GPU runtime outside the venv: `tools/setup.sh --gpu` installs `onnxruntime-gpu` 1.32.0 with
  NVIDIA's CUDA 13 and cuDNN 9 libraries into `~/.cache/xblick/yolox/onnxruntime-gpu/` once. It stays across `uv run`,
  `uv sync` and `tools/setup.sh`. The swap of packages in `.venv` that 0.1.1 documented did not survive `uv run`.
- On a GPU every part of the model runs there (ONNX Runtime's fallback to the CPU is off). Without the GPU runtime
  the start ends with one line that names `tools/setup.sh --gpu`. When CUDA cannot be loaded, the line names the
  libraries that failed.
- With `--device cuda` (GPU 0) ONNX Runtime no longer prints the warning "No registered plugin EP device found".
- The start line says where the model computes: "model on the CPU with 4 threads" or "model on cuda:0; scaling,
  decoding and drawing on the CPU".

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
