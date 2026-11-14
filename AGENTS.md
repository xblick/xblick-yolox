# AGENTS.md

xblick-yolox: Megvii's YOLOX object detector (COCO, 80 classes) as a Durchblick node on the CPU. ONNX Runtime runs
the ONNX files of YOLOX's release 0.1.1rc0; pre- and post-processing in numpy (`model.py`). Optional overlay image
stream (`--overlay`). Version 0.1.3, tested on a test camera and a published image, not with a real webcam, not on a
GPU. Human documentation: `README.md`, `docs/design.md`, `docs/performance.md`. Transport-wide traps are in the core's
`AGENTS.md` (github.com/xblick/xblick). Style of all documentation: the core's `docs/project/writing-docs.md`.

## Commands

```bash
tools/setup.sh                                       # uv sync --frozen, then xblick-yolox --download-only (tiny)
.venv/bin/python -m pytest -q                        # ~7 s; real-model tests skip without the models in ~/.cache/xblick/yolox
.venv/bin/xblick-yolox --download-only --model s     # one model into ~/.cache/xblick/yolox
.venv/bin/ruff check src tests tools                 # lint; 120 columns
taskset -c 0-3 .venv/bin/python tools/bench.py       # ms per frame of each model in ~/.cache/xblick/yolox, ~1 min
.venv/bin/xblick router & .venv/bin/xblick test-camera &   # router and camera, as the README's Run
.venv/bin/xblick-yolox --camera test/camera --overlay
```

## Rules

- Files: the weights are in the node's cache directory (`xblick.paths.cache_dir("yolox")`, `~/.cache/xblick/yolox`), never in
  the checkout. `weights.STORE = Store(PROG, node=NODE, ..., former=[FORMER])`, `Weight` paths relative to its root;
  `store = weights.store_for(parser, args)` (`--weights-dir`) for `download_only`, `ensure` and `fail`, with
  `options=weights.options(args)`; model code takes its paths from `store.root`. `FORMER` (the checkout's `weights/`) is never
  read: the store prints the command that moves it. The core's `docs/reference/files.md` has the rule.
- Fixed defaults only: `--model tiny`, `--score 0.3`, `--nms 0.45`, `--threads 4`, `--device cpu`. Never derive a
  default from the machine, the installed packages or a measurement.
- `--device cuda[:N]` without a usable CUDA provider is an error. Never fall back to the CPU.
- Overlay: the core's `xblick.overlay` (`add_argument`, `Overlay`, `draw_boxes`). No `Overlay` without `--overlay`.
  Publish the detections before `overlay.put`. Report what the core helper lacks; do not work around it here.
- Weights: `weights.py` is the one list (`MODELS`: URL, size, SHA-256 measured from the downloaded files) and
  `STORE` (`xblick.weights.Store`, root `~/.cache/xblick/yolox`, the node's cache directory; `FORMER`: the checkout's `weights/`). The download itself is the core's.
- `model.py` follows YOLOX's ONNX format; keep `tests/test_model.py` green after any change of letterbox or decode.
- Durchblick pin: `xblick>=0.5.5,<0.6`. Dependencies from PyPI only.
- 120 columns. Docstrings say what, comments say why. Update `README.md` and `docs/` with every change of behaviour.

## Traps

| Trap | What happens | Fix |
| --- | --- | --- |
| GPU runtime | `onnxruntime-gpu` and `onnxruntime` both install the package `onnxruntime`; `uv run` brings the CPU one back over a swap (seen: CUDA provider library gone). | Never swap in `.venv`; `tools/setup.sh --gpu` installs into `<cache>/onnxruntime-gpu`, `gpu.onnxruntime_for` imports from there. |
| BGR input | YOLOX trained on BGR images with values 0 to 255, no normalisation. | `letterbox` flips RGB to BGR. |
| Resize | Pillow's bilinear antialiases when it shrinks; YOLOX used OpenCV's, which does not. Scores differ slightly. | Not verified against OpenCV. |
| Fixed input size | Each ONNX file has a fixed input (416 or 640). | `INPUT_SIZE`; `Detector` checks the shapes. |
| `classes` parameter | An unknown class is refused by `check_classes` (live) and by `parser.error` (start). | |
| Node tests | The stand-in runs the node through `python -c` with `node.load_detector` replaced. | `tests/standin.py` |
| SIGINT in scripts | A background job of a script has SIGINT ignored; the core restores it for `xblick` commands, not in `node.start`. | Stop the node by PID with SIGTERM. |

## Layout

| Path | Content |
| --- | --- |
| `src/xblick_yolox/model.py` | `letterbox`, `grids`, `decode`, `nms`, `postprocess`, `Detector`, `COCO_CLASSES`, `INPUT_SIZE` |
| `src/xblick_yolox/node.py` | options, the loop, publishing, `draw` (the overlay), diagnostics |
| `src/xblick_yolox/gpu.py` | the GPU runtime outside the venv: `runtime_dir`, `onnxruntime_for`, the pinned onnxruntime-gpu and CUDA libraries |
| `src/xblick_yolox/weights.py` | `MODELS`, `needed`, `path_of`, `STORE` |
| `~/.cache/xblick/yolox/` | `yolox_<model>.onnx`, `.download.lock`: the node's cache directory. The checkout's `weights/` is the former place. |
| `tests/` | `test_model.py`, `test_overlay.py`, `test_node.py`, `test_weights.py`, `standin.py`, `data/office.png` |
| `tools/` | `setup.sh`, `bench.py`, `check_references.sh` |

## Defaults

| What | Value |
| --- | --- |
| Input | `<camera>/color` (`--color`): `rgb8`, `rgba8`, `bgr8`, `bgra8`, `mono8` |
| Detections | `<camera>/yolox/detections_2d`, `vision_msgs/Detection2DArray`, class name as `class_id` |
| Overlay | `<camera>/yolox/overlay`, `rgb8`, only with `--overlay` |
| Diagnostics | `<camera>/diagnostics`, status `yolox`, once per second |
| `--max-hz` | none (the newest frame, as fast as it goes) |
| `--local` | `auto` (shared memory) |
