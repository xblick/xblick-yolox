# xblick-yolox

[![xblick core](https://img.shields.io/pypi/v/xblick?label=xblick%20core&color=blue)](https://github.com/xblick/xblick)

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
| CPU | an x86-64 PC with Linux. We measured on 4 cores of an AMD EPYC 9335. |
| GPU | none. An NVIDIA GPU is optional, see "On a GPU". |
| Durchblick | 0.5.3 or newer, below 0.6 (Python package `xblick`) |
| Python | 3.10 to 3.13, which `tools/setup.sh` puts into `.venv` |
| Tools for the setup | uv |
| Disk | 4 MB (nano), 20 MB (tiny) or 36 MB (s) per model |

Model licences:

| Part | Licence |
| --- | --- |
| YOLOX's code | Apache 2.0, as the repository's LICENSE states |
| ONNX model files of YOLOX's release 0.1.1rc0 | The release states no terms of its own. The repository that publishes it is under Apache 2.0. The files are a download. |

The node prints these terms before it downloads a model. This repository contains no copy of YOLOX's code or models.

## Install

```bash
tools/setup.sh
```

The script creates `.venv` exactly as `uv.lock` states, with ONNX Runtime for the CPU and Durchblick from PyPI. Then
it runs `xblick-yolox --download-only`, which downloads YOLOX-tiny into `weights/` and checks its size and SHA-256.
`tools/setup.sh --model s` downloads another model. The script downloads only what is missing.

The node also downloads a missing model at its start. A machine that must never start a download on its own starts
the node with `--no-download`. The node then ends at once when the file is missing, and it names the file.

The node is not on PyPI yet. We plan to publish it there, so that `pip install xblick-yolox` installs it.

## Run

Start a router, a camera, the node and a viewer, each in its own terminal. The camera name is an example.

```bash
.venv/bin/xblick router                                  # the Zenoh router of this PC
.venv/bin/xblick test-camera                             # a test pattern on test/camera/color
.venv/bin/xblick-yolox --camera test/camera --overlay
# on a PC with an NVIDIA GPU: add --device cuda (see "On a GPU")
.venv/bin/xblick view test/camera/color test/camera/yolox/overlay
```

`xblick view` opens VLC, or offers the browser view at `http://127.0.0.1:8090/`. The test pattern has no objects
in it. A webcam has: `xblick cam /dev/video0` publishes `cam/video0/color`, and `--camera cam/video0` takes it.
`xblick cam` needs the core's `cam` extra (`xblick[cam]`), which `.venv` does not have.

| Option | Default | Meaning |
| --- | --- | --- |
| `--camera CAMERA` | required | the base name of the camera, such as `cam/video0` |
| `--model NAME` | `tiny` | `nano` or `tiny` (input 416 x 416), or `s` (640 x 640) |
| `--score S` | 0.3 | the least score of an object. Live. |
| `--nms IOU` | 0.45 | a box goes when it overlaps a better box of its class by more than this intersection over union. Live. |
| `--classes NAME ...` | all 80 | only these COCO classes, also comma-separated: `--classes person,cup`. Live. |
| `--overlay` | off | also publish `<camera>/yolox/overlay` |
| `--threads N` | 4 | the CPU threads of ONNX Runtime for one frame |
| `--device DEVICE` | `cpu` | `cpu`, `cuda` (GPU 0) or `cuda:N`. A GPU that ONNX Runtime cannot use ends the start with one line. |
| `--max-hz HZ` | every frame | the most frames per second. The node always takes the newest frame. |
| `--local` | `auto` | how frames of a camera on this host reach the node: `auto` or `shm` |
| `--color TOPIC` | `<camera>/color` | the colour stream |
| `--node NAME` | `yolox` | the node's part of the topics |
| `--no-download` | off | end at once when the model file is missing |
| `--download-only` | off | download the missing model, check it and end. `--camera` may be left out. |

Every default is a fixed value. The node never chooses a model, a device or a thread count from the machine.

Every option is also a parameter of the node `<camera>/yolox`, which Durchblick's terminal interface (`xblick`, tab
Parameters) and `xblick param` show. `score`, `nms` and `classes` change while the node runs. The other options apply
at the next start.

## On a GPU

`uv.lock` installs ONNX Runtime for the CPU. For a GPU, replace it in `.venv` by `onnxruntime-gpu` with NVIDIA's CUDA
13 and cuDNN 9 libraries from PyPI (2.5 GB):

```bash
uv pip uninstall --python .venv/bin/python onnxruntime
uv pip install --python .venv/bin/python "onnxruntime-gpu[cuda,cudnn]==1.32.0"
.venv/bin/xblick-yolox --camera test/camera --overlay --device cuda
```

The node then runs YOLOX-tiny in 3.9 ms per frame on an RTX PRO 6000 (driver 580, CUDA 13.0). `tools/setup.sh`
installs the CPU package again, because it restores `.venv` as `uv.lock` states. A GPU that ONNX Runtime cannot use
ends the start with one line. The node never falls back to the CPU.

## Which model

The table gives the milliseconds per 640 x 480 frame for the whole detection: scaling, model, decoding and
suppression. We measured on 4 threads of an AMD EPYC 9335, the process limited to 4 cores, as the median of 200
frames (2026-11-14).

| Model | Input | ms per frame | Objects found in `tests/data/office.png` |
| --- | --- | --- | --- |
| nano | 416 x 416 | 6.6 | two chairs, six bottles, a table |
| tiny | 416 x 416 | 10.9 | two chairs, seven bottles |
| s | 640 x 640 | 30.7 | two chairs, bottles and a cup, with higher scores |

Take `nano` on a slow notebook. Take `s` when objects are small and the CPU has time. Megvii's README gives the
accuracy on COCO as mAP 25.8, 32.8 and 40.5. [Speed](docs/performance.md) has the setup and the node's rate end to
end.

## Limits

- We tested the node on a test camera and on a replayed image, as separate processes. We have not tested it with a
  real webcam.
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
