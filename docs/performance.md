# Speed

This page lists the measured times of the three models and the node's rate end to end.

## Setup

The PC has two AMD EPYC 9335 with 32 cores each. Every measurement limited the process to 4 cores
(`taskset -c 0-3`) and ONNX Runtime to 4 threads (`--threads 4`), to come close to a notebook. Other users' jobs ran
on the PC (load average about 7). The input was `tests/data/office.png`, 640 x 480. Date: 2026-11-14.

## Per frame

`tools/bench.py` times 200 frames after 20 frames of warm-up. "Detect" is the whole detection: scaling, model,
decoding and suppression.

| Model | Input | Detect, median | Detect, 90th percentile | Model alone, median |
| --- | --- | --- | --- | --- |
| nano | 416 x 416 | 6.6 ms | 7.5 ms | 4.6 ms |
| tiny | 416 x 416 | 10.9 ms | 11.1 ms | 8.5 ms |
| s | 640 x 640 | 30.7 ms | 31.1 ms | 29.5 ms |

```bash
taskset -c 0-3 .venv/bin/python tools/bench.py
```

## End to end

`xblick test-camera --fps 30` published 640 x 480 frames, and the node ran as its own process on 4 cores for 25 s.
The node reports "ms a frame" for its whole work on a frame, including the colour conversion and the publishing.

| Model | Frames per second | ms a frame in the node |
| --- | --- | --- |
| nano | 29.95 | 13 |
| tiny (with `--overlay`, no subscriber) | 29.94 | 24 |
| s | 30.18 | 32 |

All three models kept up with the camera. The node needs more time per frame than the benchmark, which runs frames
back to back. We have not found out why. One possible cause is the CPU's clock, which drops between frames.

## On a GPU

With `onnxruntime-gpu[cuda,cudnn]` 1.32.0 in place of `onnxruntime`, on one RTX PRO 6000 (driver 580, CUDA 13.0), the
whole detection took 3.4 ms (nano), 3.9 ms (tiny) and 3.6 ms (s) per frame, median of 100 frames, 4 CPU threads.
The node kept up with a replay at
10 frames per second (9 ms a frame in the node, tiny) and the test camera at 15 frames per second.

## Settings

ONNX Runtime's threads sleep between frames (`session.intra_op.allow_spinning` is `0`), so the node uses no CPU
while it waits for a frame. We have not measured the effect of this setting.
