"""YOLOX on the CPU: ONNX Runtime runs Megvii's exported model, numpy does the rest.

    detector = Detector(weights.path_of("tiny"), threads=4)
    found = detector.detect(rgb, score=0.3, nms=0.45)       # rgb: (H, W, 3) uint8
    found.boxes, found.scores, found.labels                 # x1 y1 x2 y2 in pixels of rgb; COCO class indices

The pre- and post-processing follow the input and output format of the ONNX files of YOLOX's release 0.1.1rc0 (the
format of YOLOX's ``tools/export_onnx.py`` and ``demo/ONNXRuntime``). They are written for this node in numpy; no code
of YOLOX is copied:

* input ``images``, (1, 3, S, S) float32: the frame scaled by one factor to fit S x S (its top-left corner at the
  input's), the rest grey 114, channels in BGR order, values 0 to 255 (no normalisation: YOLOX since 0.1.1rc0)
* output ``output``, (1, N, 85): per anchor point x, y, w, h (not yet decoded), objectness, 80 class scores (both
  after the sigmoid). The anchor points are the cells of three grids of strides 8, 16 and 32 over the input, each
  grid row by row: N = (S/8)^2 + (S/16)^2 + (S/32)^2 (3549 at 416, 8400 at 640)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

STRIDES = (8, 16, 32)
PAD = 114  # the grey of YOLOX's padding
INPUT_SIZE = {"nano": 416, "tiny": 416, "s": 640}  # the side of each model's square input, fixed in its ONNX file

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


@dataclass
class Detections:
    """Objects in a frame, best first: ``boxes`` (n, 4) float32 x1, y1, x2, y2 in pixels of the frame, ``scores``
    (n,) float32 (objectness times class score), ``labels`` (n,) int64 indices into COCO_CLASSES."""

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.scores)

    @staticmethod
    def empty() -> Detections:
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))


def letterbox(rgb: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """The model's input of an (H, W, 3) uint8 RGB frame: (1, 3, size, size) float32 BGR, the frame scaled by ``r``
    (bilinear, Pillow) into the top-left corner, the rest PAD; and ``r`` (input pixels per frame pixel)."""
    h, w = rgb.shape[:2]
    r = min(size / h, size / w)
    nw, nh = max(1, int(w * r)), max(1, int(h * r))  # rounded down, as YOLOX does
    scaled = rgb if (nw, nh) == (w, h) else np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
    blob = np.full((1, 3, size, size), PAD, np.float32)
    blob[0, :, :nh, :nw] = scaled[..., ::-1].transpose(2, 0, 1)  # BGR, channels first
    return blob, r


def grids(size: int) -> tuple[np.ndarray, np.ndarray]:
    """The anchor points of an input of ``size``: (N, 2) cell x, y and (N, 1) the stride of each, in the order of the
    model's output (stride 8 first, each grid row by row)."""
    cells, strides = [], []
    for stride in STRIDES:
        n = size // stride
        ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        cells.append(np.stack((xs, ys), -1).reshape(-1, 2))
        strides.append(np.full((n * n, 1), stride))
    return np.concatenate(cells).astype(np.float32), np.concatenate(strides).astype(np.float32)


def decode(raw: np.ndarray, cells: np.ndarray, strides: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The model's (N, 85) output as boxes (N, 4) x1, y1, x2, y2 in input pixels, the best class's score (N,)
    (objectness times class score) and that class (N,)."""
    centre = (raw[:, :2] + cells) * strides
    half = np.exp(raw[:, 2:4]) * strides / 2
    boxes = np.concatenate((centre - half, centre + half), 1)
    classes = raw[:, 5:].argmax(1)
    scores = raw[:, 4] * raw[np.arange(len(raw)), 5 + classes]
    return boxes, scores, classes


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Greedy non-maximum suppression: the indices kept, best first. A box goes when it overlaps a better kept box by
    more than ``iou`` (intersection over union)."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        union = area[i] + area[rest] - inter
        overlap = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[overlap <= iou]
    return np.asarray(keep, np.int64)


def postprocess(raw: np.ndarray, cells: np.ndarray, strides: np.ndarray, r: float, frame_size: tuple[int, int],
                score: float, iou: float, classes: set[int] | None = None) -> Detections:
    """Detections of one frame from the model's (N, 85) output: the anchor points whose best class scores at least
    ``score`` (and is one of ``classes``, None: all), non-maximum suppression per class (a box suppresses only boxes of
    its own class), boxes in frame pixels (divided by ``r``, clipped to the frame of ``frame_size`` (w, h))."""
    boxes, scores, labels = decode(raw, cells, strides)
    chosen = scores >= score
    if classes is not None:
        chosen &= np.isin(labels, list(classes))
    boxes, scores, labels = boxes[chosen], scores[chosen], labels[chosen]
    if not len(scores):
        return Detections.empty()
    # one NMS for all classes: each class's boxes moved apart by more than any box is wide
    offset = labels[:, None].astype(np.float32) * (boxes.max() - boxes.min() + 1.0)
    kept = nms(boxes + offset, scores, iou)
    w, h = frame_size
    boxes = boxes[kept] / r
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)
    return Detections(boxes.astype(np.float32), scores[kept].astype(np.float32), labels[kept].astype(np.int64))


class Detector:
    """One YOLOX model in ONNX Runtime, ``threads`` CPU threads for one frame at a time; ``device`` ``cpu``, or
    ``cuda:N`` (ONNX Runtime's CUDA provider on GPU N, which the onnxruntime-gpu package brings: without it an
    error, never the CPU instead)."""

    def __init__(self, path: Path, size: int, threads: int, device: str = "cpu"):
        self.size = size
        self.session = self._load(path, threads, device)
        self.cells, self.strides = grids(size)

    def _load(self, path: Path, threads: int, device: str):
        import onnxruntime as ort  # noqa: PLC0415 - loaded with the model

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads  # the threads for one model call
        options.inter_op_num_threads = 1  # one call at a time
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # the threads sleep between frames (a notebook's battery) instead of spinning while they wait for work
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        providers = ["CPUExecutionProvider"]
        if device != "cpu":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError(f"--device {device}: this onnxruntime has no CUDA provider (it has "
                                   f"{', '.join(ort.get_available_providers())}); install onnxruntime-gpu in its "
                                   "place, or use --device cpu")
            # CUDA and cuDNN of the nvidia-* wheels that onnxruntime-gpu[cuda,cudnn] installs (else the system's)
            ort.preload_dlls()
            providers = [("CUDAExecutionProvider", {"device_id": int(device.split(":")[1])}), "CPUExecutionProvider"]
        session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        if device != "cpu" and session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"--device {device}: ONNX Runtime could not use the GPU ({session.get_providers()})")
        (given,), (out,) = session.get_inputs(), session.get_outputs()
        anchors = sum((self.size // s) ** 2 for s in STRIDES)
        if list(given.shape) != [1, 3, self.size, self.size] or list(out.shape) != [1, anchors, 5 + len(COCO_CLASSES)]:
            raise ValueError(f"{path}: input {given.shape} and output {out.shape}, expected [1, 3, {self.size}, "
                             f"{self.size}] and [1, {anchors}, 85] (a YOLOX ONNX file of COCO's 80 classes)")
        self._input = given.name
        if device != "cpu":  # a GPU that loads but cannot compute (no cuDNN) fails here, at the start
            try:
                session.run(None, {given.name: np.zeros((1, 3, self.size, self.size), np.float32)})
            except Exception as error:  # noqa: BLE001 - ONNX Runtime's errors: one line for the user
                raise RuntimeError(f"--device {device}: ONNX Runtime failed on the GPU: {str(error)[:300]}") from None
        return session

    def infer(self, blob: np.ndarray) -> np.ndarray:
        """The model's (N, 85) output for one input blob."""
        return self.session.run(None, {self._input: blob})[0][0]

    def detect(self, rgb: np.ndarray, score: float, nms: float, classes: set[int] | None = None) -> Detections:
        """The objects in an (H, W, 3) uint8 RGB frame (see ``postprocess``)."""
        blob, r = letterbox(rgb, self.size)
        raw = self.infer(blob)
        return postprocess(raw, self.cells, self.strides, r, (rgb.shape[1], rgb.shape[0]), score, nms, classes)
