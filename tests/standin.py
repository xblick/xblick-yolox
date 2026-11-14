"""Synthetic YOLOX outputs, for the tests: ``raw_for`` builds the (N, 85) output in which the given boxes are
found, and ``StandIn`` is a Detector that returns it without a model (no ONNX Runtime, no weights)."""

import numpy as np

from xblick_yolox import model

# boxes in input pixels of a 416 input (x1, y1, x2, y2), class, score: a person and a cup
OBJECTS = [((40.0, 60.0, 200.0, 380.0), 0, 0.9), ((260.0, 200.0, 300.0, 250.0), 41, 0.6)]


def raw_for(objects, size: int) -> np.ndarray:
    """The output in which each object is the anchor point of stride 32 whose cell holds its centre."""
    cells, strides = model.grids(size)
    raw = np.zeros((len(cells), 85), np.float32)
    raw[:, 2:4] = -5.0  # tiny boxes, objectness 0
    for (x1, y1, x2, y2), label, score in objects:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        i = int(np.flatnonzero((strides[:, 0] == 32) & (cells[:, 0] == cx // 32) & (cells[:, 1] == cy // 32))[0])
        raw[i, :2] = (cx / 32 - cells[i, 0], cy / 32 - cells[i, 1])
        raw[i, 2:4] = np.log(((x2 - x1) / 32, (y2 - y1) / 32))
        raw[i, 4] = 1.0
        raw[i, 5 + label] = score
    return raw


class StandIn(model.Detector):
    def _load(self, path, threads, device):
        self.size = 416
        return None

    def infer(self, blob):
        assert blob.shape == (1, 3, self.size, self.size) and blob.dtype == np.float32
        return raw_for(OBJECTS, self.size)
