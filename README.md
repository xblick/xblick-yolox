# xblick-yolox

[![xblick core](https://img.shields.io/pypi/v/xblick?label=xblick%20core&color=blue)](https://github.com/xblick/xblick)
[![tests](https://github.com/xblick/xblick-yolox/actions/workflows/tests.yml/badge.svg)](https://github.com/xblick/xblick-yolox/actions/workflows/tests.yml)
[![licence](https://img.shields.io/badge/licence-Apache--2.0-blue)](LICENSE)
[![device](https://img.shields.io/badge/runs%20on-CPU%20or%20NVIDIA%20GPU-brightgreen)](README.md)

This [Durchblick](https://github.com/xblick/xblick) node finds the objects of COCO's 80 classes in a camera's colour
stream. It runs on a CPU without a GPU: YOLOX-tiny needs 11 ms per 640 x 480 frame on 4 CPU threads of our test PC,
and the node kept up with a camera at 30 frames per second.

The node uses [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) by Megvii (2021), a one-stage object detector,
in the ONNX files of YOLOX's release. ONNX Runtime runs the model, and numpy does the pre- and post-processing.

## What it publishes

The node publishes for the newest colour frame, as often as the CPU finishes one. Each message has the stamp and the
camera frame of its colour frame.

| Topic | Message type | Content |
| --- | --- | --- |
| `<camera>/yolox/detections_2d` | `vision_msgs/Detection2DArray` | one detection per object, the best first. `class_id` is the class name, such as `person`, and `score` is objectness times class score. The box is in pixels of the colour image. |
| `<camera>/yolox/overlay` | image stream, encoding `rgb8` | only with `--overlay`: the colour image with the boxes, class names and scores drawn on it |
| `<camera>/diagnostics` | `diagnostic_msgs/DiagnosticArray` | status `yolox` once per second: frames per second, ms per frame, objects per frame, ms from the frame to the results |

A frame without objects gives an empty array. `--node` changes the `yolox` part of the topics. The node reads
`<camera>/color` in the encoding `rgb8`, `rgba8`, `bgr8`, `bgra8` or `mono8`.

The overlay is off by default, so that the topics hold only results. With `--overlay` the node draws in a thread of
its own after it has published the detections, so the drawing never delays them. It draws only while a subscriber
exists. Without `--overlay` the stream does not exist at all.
[How the node works](docs/design.md#the-overlay) describes the rules.

## Requirements

| What | Requirement |
| --- | --- |
| CPU | an x86-64 PC with Linux |
| GPU | none. An NVIDIA GPU is optional, see "On a GPU". |
| Durchblick | 0.5.3 or newer, below 0.6 (Python package `xblick`) |
| Python | 3.10 to 3.13, which uv installs with the node |
| Tools for the install | git and uv (step 1 of the installation installs them) |
| Disk | 4 MB (nano) to 396 MB (x) per model, see "Which model" |

Model licences:

| Part | Licence |
| --- | --- |
| YOLOX's code | Apache 2.0, as the repository's LICENSE states |
| ONNX model files of YOLOX's release 0.1.1rc0, all seven | The release states no terms of its own, for none of the files. The repository that publishes them is under Apache 2.0. All seven are files of that one release, so no model has other terms. The files are a download. |

The node prints these terms before it downloads a model. This repository contains no copy of YOLOX's code or models.

## Install

1. Install the prerequisites. After the installation of uv, open a new terminal, so that `uv` is on `PATH`.

   ```bash
   sudo apt install git
   wget -qO- https://astral.sh/uv/install.sh | sh
   ```

   A GPU is optional. For `--device cuda`, `nvidia-smi` must print a driver 580 or newer. On Ubuntu,
   `sudo apt install nvidia-driver-580` and a restart install that driver.

2. Install the node and download its model:

   ```bash
   uv tool install git+https://github.com/xblick/xblick-yolox
   xblick-yolox --download-only
   ```

   uv puts `xblick-yolox` on `PATH`, in an environment of its own (about 0.2 GB), with ONNX Runtime for the CPU and a
   Python 3.10 to 3.13 that uv finds or downloads. It resolves the dependencies afresh within the bounds of
   `pyproject.toml`: `uv.lock` applies only to a clone. `xblick-yolox --download-only` prints the model's terms,
   downloads YOLOX-tiny (20 MB) into `~/.cache/xblick/yolox/` and checks its size and SHA-256. `--model s`
   downloads another model. For `--device cuda`, add the GPU runtime as [On a GPU](#on-a-gpu) shows.

   `uv tool upgrade xblick-yolox` updates the node to the newest commit of this repository and the dependencies
   within their bounds. `uv tool uninstall xblick-yolox` removes the node. The models and the GPU runtime stay in
   the cache directory.

3. Install the core's command `xblick` with the viewer, video files and webcams. uv puts it on `PATH`. A second
   `uv tool install` of xblick with other extras replaces the first and removes what only the first had, so every
   xblick node names this one line:

   ```bash
   uv tool install "xblick[rerun,video,cam]>=0.5,<0.6"
   ```

4. Start the node on photos, without a camera. Take a few JPEG photos of rooms or tables with everyday objects, for
   example in `~/Pictures`. Run each command in its own terminal. If the PC already runs a Zenoh router (`xblick
   router` or `zenohd`), leave out the first one.

   ```bash
   xblick router                                         # the Zenoh router of this PC
   xblick play "$HOME/Pictures/*.jpg" --camera demo/photos --fps 1 --loop
   xblick-yolox --camera demo/photos --overlay
   xblick view demo/photos/color demo/photos/yolox/overlay
   ```

   `xblick play` shows the photos one per second as the camera `demo/photos`. `xblick view` opens both streams in
   the browser: each photo, and beside it the same photo with a box, a class name and a score on every object that
   YOLOX finds among its 80 COCO classes, such as person, cup, chair or bottle.

### From a clone

For development, and for the versions that `uv.lock` fixes:

```bash
git clone https://github.com/xblick/xblick-yolox.git
cd xblick-yolox
tools/setup.sh
```

When git or uv is missing, the script stops and prints the command that installs it. `tools/setup.sh --help` lists
the options. In a clone the commands are in `.venv/bin/`.

On a PC with an NVIDIA GPU the script also installs the GPU runtime for `--device cuda` (2.4 GB), unless you add
`--no-gpu`. [On a GPU](#on-a-gpu) describes the rule.

The script creates `.venv` exactly as `uv.lock` states, with ONNX Runtime for the CPU and Durchblick from PyPI. Then
it runs `xblick-yolox --download-only`, which downloads YOLOX-tiny into `~/.cache/xblick/yolox/` and checks its size and SHA-256.
`tools/setup.sh --model s` downloads another model. The script downloads only what is missing.

### Downloads at the start

The node also downloads a missing model at its start. A machine that must never start a download on its own starts
the node with `--no-download`. The node then ends at once when the file is missing, and it names the file.

The node is not on PyPI yet. We plan to publish it there, so that `pip install xblick-yolox` installs it.

### Where the weights are

The node keeps its weights in its cache directory, `~/.cache/xblick/yolox/`. `XBLICK_HOME=/dir` moves it to `/dir/cache/yolox/`,
and `XDG_CACHE_HOME` moves it as well. `--weights-dir DIR` names another directory. The place is the same for an
install with uv and for a clone. [Where files are](https://github.com/xblick/xblick/blob/main/docs/reference/files.md) lists the places, and it shows how to copy the cache to a
PC without internet.

Up to version 0.1.1, the weights were in the checkout's `weights/`. The node does not read them there. At its start it
prints the command that moves them. In the checkout it is:

```bash
mkdir -p ~/.cache/xblick && mv weights ~/.cache/xblick/yolox
```

## Run

The node reads the camera that `--camera` names. A webcam works without other nodes: `xblick cam /dev/video0`
publishes `cam/video0/color`, and `--camera cam/video0` takes it. In a clone the commands are in `.venv/bin/`.

```bash
xblick cam /dev/video0
xblick-yolox --camera cam/video0 --overlay
# on a PC with an NVIDIA GPU: add --device cuda (see "On a GPU")
xblick view cam/video0/color cam/video0/yolox/overlay
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--camera CAMERA` | required | the base name of the camera, such as `cam/video0` |
| `--model NAME` | `tiny` | `nano`, `tiny` (input 416 x 416), `s`, `m`, `l`, `x` or `darknet53` (640 x 640), see "Which model" |
| `--score S` | 0.3 | the least score of an object. Live. |
| `--nms IOU` | 0.45 | a box goes when it overlaps a better box of its class by more than this intersection over union. Live. |
| `--classes NAME ...` | all 80 | only these COCO classes, also comma-separated: `--classes person,cup`. Live. |
| `--overlay` | off | also publish `<camera>/yolox/overlay` |
| `--threads N` | 4 | the CPU threads of ONNX Runtime for one frame on the CPU |
| `--device DEVICE` | `cpu` | `cpu`, `cuda` (GPU 0) or `cuda:N`. A GPU that ONNX Runtime cannot use ends the start with one line. |
| `--max-hz HZ` | every frame | the most frames per second. The node always takes the newest frame. |
| `--local` | `auto` | how frames of a camera on this host reach the node: `auto` or `shm` |
| `--color TOPIC` | `<camera>/color` | the colour stream |
| `--node NAME` | `yolox` | the node's part of the topics |
| `--no-download` | off | end at once when the model file is missing |
| `--download-only` | off | download the missing model, check it and end. `--camera` may be left out. |
| `--weights-dir DIR` | the cache directory | where the weights are and go, in place of `~/.cache/xblick/yolox/` |

Every default is a fixed value. The node never chooses a model, a device or a thread count from the machine.

Every option is also a parameter of the node `<camera>/yolox`, which Durchblick's terminal interface (`xblick`, tab
Parameters) and `xblick param` show. `score`, `nms` and `classes` change while the node runs. The other options apply
at the next start.

## On a GPU

ONNX Runtime for a GPU is a package of its own, `onnxruntime-gpu`, which cannot sit in `.venv` beside the CPU
package: both install the Python package `onnxruntime`, and `uv run` and `uv sync` bring the CPU package back. The
GPU runtime therefore lives outside `.venv`, in one directory that every Durchblick node with ONNX Runtime shares:

```
~/.cache/xblick/onnxruntime/gpu-1.32.0-cp312/      (onnxruntime-gpu version, Python version)
```

`tools/setup.sh` installs it when the PC has an NVIDIA GPU. The rule: `nvidia-smi` runs, exits 0 and lists at least
one GPU. Before the download the script prints the GPUs it found, what it installs and where, and how to skip it. The
runtime is `onnxruntime-gpu` 1.32.0 with NVIDIA's CUDA 13 and cuDNN 9 libraries from PyPI (2.4 GB). A second call, or
another node that installed it already, reuses it. `uv run`, `uv sync` and `tools/setup.sh` leave it as it is.

| Case | What `tools/setup.sh` does |
| --- | --- |
| no `nvidia-smi`, or it fails or lists no GPU | one line: the node computes on the CPU, `tools/setup.sh --gpu` adds the runtime later |
| a GPU with a driver older than 580 | one line: the runtime needs driver 580 or newer and was not installed |
| a GPU with driver 580 or newer | installs the runtime, or reuses it |
| `--gpu` | installs the runtime whatever the check says (an image built on a PC without a GPU) |
| `--no-gpu` | never installs it |

After an install with uv, which has no `tools/setup.sh`, the node's second command installs the runtime. It
checks for no GPU first, so run it on a PC whose `nvidia-smi` prints a driver 580 or newer:

```bash
xblick-yolox-gpu-runtime install
```

Requirements of the GPU runtime: an NVIDIA driver 580 or newer (CUDA 13), a GPU that CUDA 13 supports (Turing, RTX
20 series, or newer), Python 3.11 or newer (onnxruntime-gpu 1.32.0 has no wheel for 3.10). NVIDIA's libraries come as
wheels, so the PC needs no CUDA installation.

```bash
xblick-yolox --camera demo/photos --overlay --device cuda
```

With `--device cuda` the node loads ONNX Runtime from the shared directory, and every part of the model runs on the
GPU. A model that the GPU cannot run whole is an error, never a silent switch to the CPU. Scaling the frame, decoding the boxes, the suppression and the overlay's drawing stay on the CPU. The start
prints "preparing the GPU" at once, then the start line, for example "model on cuda:0; ... on the CPU". Loading CUDA
and the first computation can take a while on the first start. Ctrl-C and SIGTERM end the node during that time too.

Version 0.1.3 kept the runtime in `~/.cache/xblick/yolox/onnxruntime-gpu/`. The node does not use it there, and both
the node and `tools/setup.sh` print the `mv` to the shared place.

Without the GPU runtime, `--device cuda` ends the start with one line that names `xblick-yolox-gpu-runtime
install` and, for a clone, `tools/setup.sh --gpu`. When CUDA
cannot be loaded, for example with an older driver, the line gives ONNX Runtime's message and the libraries that failed
to load. With `--device cuda:1` or another GPU than 0, ONNX Runtime 1.32 prints one warning, "No registered plugin EP
device found for 'CUDAExecutionProvider' with device_id=1". It is harmless: ONNX Runtime then uses its CUDA provider
as usual.

## Which model

All seven models of YOLOX's release are offered. Input, parameters, FLOPs and mAP are YOLOX's own figures from
[its benchmark tables](https://github.com/Megvii-BaseDetection/YOLOX#benchmark) (mAP on COCO val2017). The file size
is that of the ONNX file. The milliseconds are ours: the whole detection of a 640 x 480 frame on 4 threads of an AMD
EPYC 9335, the process limited to 4 cores, median of 200 frames (2026-11-14). We measured only nano, tiny and s.

| Model | Input | Parameters | FLOPs | mAP | ONNX file | ms per frame (ours, CPU) |
| --- | --- | --- | --- | --- | --- | --- |
| nano | 416 x 416 | 0.91 M | 1.08 G | 25.8 | 3.7 MB | 6.6 |
| tiny | 416 x 416 | 5.06 M | 6.45 G | 32.8 | 20.2 MB | 10.9 |
| s | 640 x 640 | 9.0 M | 26.8 G | 40.5 | 35.9 MB | 30.7 |
| m | 640 x 640 | 25.3 M | 73.8 G | 46.9 | 101.3 MB | not measured |
| l | 640 x 640 | 54.2 M | 155.6 G | 49.7 | 216.7 MB | not measured |
| x | 640 x 640 | 99.1 M | 281.9 G | 51.1 | 396.1 MB | not measured |
| darknet53 | 640 x 640 | 63.7 M | 185.3 G | 47.7 | 254.8 MB | not measured |

On a notebook's CPU, nano and tiny are realistic for a live camera by our numbers. s took 31 ms per frame on 4
server cores. m, l, x and darknet53 need 3 to 11 times the FLOPs of s, so they are for a GPU. `darknet53` is
YOLOX-Darknet53, YOLOv3's backbone with YOLOX's head. Its release file is `yolox_darknet.onnx`.

## Limits

- In a shell script, Ctrl-C (SIGINT) does not end the node when the script started it in the background. SIGTERM
  does. The core restores SIGINT for its own commands, but not for nodes.
- YOLOX knows only COCO's 80 classes. It does not track, so a detection has no id.
- The node runs on Linux on x86-64, because the core's wheels exist only there.

## Documentation

- [How the node works](docs/design.md): the model's input and output, the post-processing, the overlay, the tests
- [Speed](docs/performance.md): the measurements and their setup
- Durchblick: [streams and topics](https://github.com/xblick/xblick/blob/main/docs/concepts/streams-and-topics.md),
  [sessions and routers](https://github.com/xblick/xblick/blob/main/docs/concepts/sessions.md)

## Licence and credits

YOLOX is by Megvii ([Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)), under the Apache
License 2.0. "Requirements" lists what its release states about the models.

The code of this repository is under the Apache License 2.0, see [LICENSE](LICENSE). [NOTICE](NOTICE) names the
copyright holder and the parts of others, among them the test image of the TUM RGB-D dataset (CC BY 4.0).

## Citation

This node uses YOLOX (Megvii-BaseDetection/YOLOX). If you use it in research, cite their work. The entries below are
copied from [its README](https://github.com/Megvii-BaseDetection/YOLOX/blob/0.1.1rc0/README.md):

```bibtex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
