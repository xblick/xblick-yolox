"""Milliseconds a frame of each YOLOX model on this PC's CPU: the whole detect (letterbox, model, decode, NMS) on
tests/data/office.png (640 x 480), after a warm-up; the median and the 90th percentile.

    taskset -c 0-3 .venv/bin/python tools/bench.py [--threads 4] [--frames 200]
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from xblick_yolox import model, weights

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--frames", type=int, default=200)
args = parser.parse_args()
rgb = np.asarray(Image.open(Path(__file__).parents[1] / "tests" / "data" / "office.png").convert("RGB"))
print(f"{rgb.shape[1]} x {rgb.shape[0]}, {args.threads} threads, {args.frames} frames after 20")
for name in weights.MODELS:
    detector = model.Detector(weights.path_of(name), model.INPUT_SIZE[name], args.threads)
    for _ in range(20):
        detector.detect(rgb, 0.3, 0.45)
    total, inner = [], []
    for _ in range(args.frames):
        t0 = time.perf_counter()
        blob, _ = model.letterbox(rgb, detector.size)
        t1 = time.perf_counter()
        detector.infer(blob)
        t2 = time.perf_counter()
        detector.detect(rgb, 0.3, 0.45)
        t3 = time.perf_counter()
        total.append((t3 - t2) * 1e3)
        inner.append((t2 - t1) * 1e3)
    q = lambda xs, p: float(np.percentile(xs, p))  # noqa: E731
    print(f"{name:5s} {model.INPUT_SIZE[name]}: detect {statistics.median(total):6.1f} ms (p90 {q(total, 90):.1f}), "
          f"model alone {statistics.median(inner):6.1f} ms")
