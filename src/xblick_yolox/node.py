"""xblick-yolox: YOLOX's objects of COCO's 80 classes (model.py) in a camera's colour stream, on the CPU.

    xblick-yolox --camera cam/video0 [--model nano|tiny|s] [--score 0.3] [--nms 0.45] [--classes person,cup] \\
                 [--overlay] [--threads 4] [--device cpu|cuda|cuda:N] [--max-hz HZ] \\
                 [--color <camera>/color] [--node yolox] [--no-download]
                 [--connect ...] [--listen ...] [--config FILE]   (default: the PC's Zenoh router)

For the newest colour frame, as often as the CPU gets to it and at most --max-hz frames a second, it publishes,
stamped as the frame and in its camera frame:

* ``<camera>/<node>/detections_2d``  vision_msgs/Detection2DArray: one detection per object, best first, its class
  name as class_id with its score (objectness times class score), its box in pixels of the colour image; no id (YOLOX
  does not track). A frame without objects publishes an empty array.
* ``<camera>/<node>/overlay``        only with --overlay: an image stream, rgb8, the colour frame with the boxes,
  class names and scores drawn on it, for any image viewer. Drawn in a thread of its own after the detections are
  published, only while somebody subscribes. Without --overlay the stream does not exist.

Frames per second, ms a frame, objects per frame and the time from frame to results go to ``<camera>/diagnostics``
(status ``yolox``) once a second.

Weights (weights.py): a missing model file is downloaded at the start, before the model loads; --no-download ends
the node instead. ``xblick-yolox --download-only [--model tiny]`` only downloads and checks it (no camera, no Zenoh).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np
import xblick as db
from xblick import node, topics
from xblick.camera import check_local
from xblick.history import FrameHistory
from xblick.session import add_arguments, open_from_args

from . import weights
from .model import COCO_CLASSES, INPUT_SIZE

COLOR_ENCODINGS = ("rgb8", "rgba8", "bgr8", "bgra8", "mono8")
KEEP_FRAMES = 2  # frames kept until taken: the node always takes the newest


def to_rgb(image: np.ndarray, encoding: str) -> np.ndarray:
    """A colour frame's image as (H, W, 3) uint8 RGB (a grey frame in all three channels)."""
    if encoding not in COLOR_ENCODINGS:
        raise ValueError(f"a colour frame of encoding {encoding!r}; one of {', '.join(COLOR_ENCODINGS)} expected")
    if encoding == "mono8":
        grey = image if image.ndim == 2 else image[..., 0]
        return np.repeat(grey[..., None], 3, axis=2)
    rgb = image[..., :3]
    return np.ascontiguousarray(rgb[..., ::-1] if encoding.startswith("bgr") else rgb)


def split_classes(given: list[str]) -> list[str]:
    """Class names as the option takes them: each also comma-separated (``person,cup`` or ``person "cell phone"``)."""
    return [s.strip() for name in given for s in name.split(",") if s.strip()]


def class_indices(names: list[str]) -> set[int] | None:
    """The COCO indices of ``names``; None for no names (every class). ValueError naming an unknown one."""
    unknown = [n for n in names if n not in COCO_CLASSES]
    if unknown:
        raise ValueError(f"no COCO class: {', '.join(unknown)} (the 80 classes: {', '.join(COCO_CLASSES)})")
    return {COCO_CLASSES.index(n) for n in names} or None


def to_message(found, header: db.Header) -> db.Detection2DArray:
    return db.Detection2DArray(header, [
        db.Detection2D(header, [db.ObjectHypothesisWithPose(db.ObjectHypothesis(COCO_CLASSES[int(label)], float(s)))],
                       db.BoundingBox2D(db.Pose2D(db.Point2D(float(x1 + x2) / 2, float(y1 + y2) / 2), 0.0),
                                        float(x2 - x1), float(y2 - y1)))
        for (x1, y1, x2, y2), s, label in zip(found.boxes, found.scores, found.labels, strict=True)])


def check_classes(value) -> str | None:
    """The live parameter classes' writer: why a list is refused (an unknown class), else None."""
    try:
        class_indices(split_classes(list(value)))
    except ValueError as error:
        return str(error)
    return None


# the parameters that change while the node runs (Parameters.declare_options); every other option applies at the start
LIVE = {"score": {"range": (0.0, 1.0, 0.01)},
        "nms": {"range": (0.0, 1.0, 0.01)},
        "classes": {"write": check_classes}}


def parse_device(text: str) -> str:
    """``--device``: cpu, cuda (GPU 0) or cuda:N, as ``cpu`` or ``cuda:N``."""
    if text == "cpu":
        return text
    if text == "cuda":
        return "cuda:0"
    if text.startswith("cuda:") and text[5:].isdigit():
        return text
    raise argparse.ArgumentTypeError(f"{text!r}: cpu, cuda (GPU 0) or cuda:N")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xblick-yolox", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", required=True, help="the camera's base, e.g. cam/video0 or test/camera")
    parser.add_argument("--model", choices=tuple(weights.MODELS), default="tiny",
                        help="the YOLOX model: nano (416 x 416 input), tiny (416 x 416) or s (640 x 640); "
                             "downloaded at the first start (default: tiny)")
    parser.add_argument("--score", type=float, default=0.3,
                        help="the least score of an object, objectness times class score (default: 0.3)")
    parser.add_argument("--nms", type=float, default=0.45,
                        help="non-maximum suppression: a box goes when it overlaps a better one of its class by more "
                             "than this intersection over union (default: 0.45)")
    parser.add_argument("--classes", nargs="*", default=[],
                        help="only these COCO classes, each also comma-separated, e.g. person,cup or person "
                             "\"cell phone\" (default: all 80)")
    parser.add_argument("--overlay", action="store_true",
                        help="also publish <camera>/<node>/overlay: the colour image with the detections drawn on it "
                             "(default: off, the stream does not exist)")
    parser.add_argument("--threads", type=int, default=4,
                        help="CPU threads for the model (ONNX Runtime's intra-op threads) (default: 4)")
    parser.add_argument("--device", type=parse_device, default="cpu",
                        help="cpu, or cuda (GPU 0) or cuda:N with the onnxruntime-gpu package; a GPU that cannot be "
                             "used is an error (default: cpu)")
    parser.add_argument("--color", default=None, help="the colour stream (default <camera>/color)")
    parser.add_argument("--node", default="yolox", help="the results' part of the topic: <camera>/<node>/...")
    weights.add_arguments(parser)  # --download-only (a mode of the command line), --no-download
    add_arguments(parser)  # --connect, --listen, --config; default: the PC's Zenoh router
    # --max-hz: every frame it can take by default; --local: auto or shm (a CPU node reads shared memory)
    node.add_arguments(parser, local=("auto", "shm"))
    node.Parameters.add_arguments(parser)  # --param NAME:=VALUE, --params-file, --dump-params, --params-read-only
    return parser


def load_detector(path, size: int, threads: int, device: str):
    """The model (model.Detector); tests replace this function by a stand-in."""
    from .model import Detector  # noqa: PLC0415 - loads ONNX Runtime

    return Detector(path, size, threads, device)


def main(argv=None) -> int:
    parser = make_parser()
    args = weights.parse_args(parser, argv)  # --download-only: --camera not required
    if args.download_only:
        return weights.download_only(weights.needed(args.model))
    camera = topics.check_name(args.camera)
    # the options as parameters (the terminal interface shows them): read-only, but those LIVE, which the loop reads
    session = open_from_args(args)
    params = node.Parameters(session, args.node, camera)
    params.configure(args)
    params.declare_options(parser, args, live=LIVE, leave_out=("camera", "node", "download_only"))
    params.check_overrides()
    problem = None
    if not 0.0 <= args.score <= 1.0 or not 0.0 <= args.nms <= 1.0:
        problem = "--score and --nms: from 0 to 1"
    elif args.threads < 1:
        problem = "--threads: at least 1"
    else:
        try:
            class_indices(split_classes(args.classes))
        except ValueError as error:
            problem = f"--classes: {error}"
    if problem is not None:
        params.close()
        session.close()
        parser.error(problem)
    color_base = args.color or f"{camera}/color"

    node.start("xblick-yolox", max_hz=args.max_hz, local=args.local)
    try:
        weights.ensure(weights.needed(args.model), download=not args.no_download)
    except weights.WeightsError as error:
        params.close()
        session.close()
        return weights.fail(error)
    try:
        detector = load_detector(weights.path_of(args.model), INPUT_SIZE[args.model], args.threads, args.device)
    except RuntimeError as error:  # a --device that ONNX Runtime cannot use: one line, no CPU instead
        print(f"xblick-yolox: {error}", file=sys.stderr, flush=True)
        params.close()
        session.close()
        return 1

    from .overlay import OverlayWorker  # noqa: PLC0415 - Pillow's drawing, only with --overlay

    wake = threading.Event()
    color = db.ImageSubscriber(session, color_base, transport="auto", local=args.local, max_hz=args.max_hz)
    history = FrameHistory(color, KEEP_FRAMES, wake.set)
    detections_pub = db.DetectionPublisher(session, topics.detections(camera, args.node, "detections_2d"),
                                           db.Detection2DArray)
    overlay = None
    if args.overlay:  # without the option no publisher exists: no topic, no info record
        overlay = OverlayWorker(db.CameraPublisher(session, f"{camera}/{args.node}/overlay", encoding="rgb8",
                                                   frame_id="", source="yolox"), COCO_CLASSES)
    report = node.Reporter(session, camera, args.node, "xblick-yolox", max_hz=args.max_hz)
    report.watch(color)
    outputs = "detections_2d" + (",overlay" if overlay is not None else "")
    print(f"xblick-yolox: {color_base} -> {camera}/{args.node}/{{{outputs}}}: YOLOX-{args.model} "
          f"({INPUT_SIZE[args.model]} x {INPUT_SIZE[args.model]}) on {args.device}, {args.threads} CPU threads",
          flush=True)
    counts = report.counts
    last = None  # the stamp of the last frame processed
    applied = (None, None)  # (classes as given, their indices)
    try:
        while True:
            wake.wait(0.05)
            wake.clear()
            history.check()
            if overlay is not None:
                overlay.check()
            stamp, frame = history.newest() or (None, None)
            if frame is None or stamp == last:
                pass
            else:
                last = stamp
                check_local(color, args.local)
                started = time.monotonic()
                given = split_classes(list(params["classes"]))
                if given != applied[0]:
                    applied = (given, class_indices(given))
                rgb = to_rgb(frame.image, frame.encoding)
                found = detector.detect(rgb, params["score"], params["nms"], applied[1])
                header = db.Header(frame.header.stamp, frame.header.frame_id)
                detections_pub.publish(to_message(found, header))  # first the results, then the overlay
                counts["busy"] += time.monotonic() - started
                counts["frames"] += 1
                counts["objects"] += len(found)
                counts["latency"] += time.time() - stamp / 1e9
                if overlay is not None:
                    overlay.put(stamp, frame.header.frame_id, rgb, found)
            if report.due():
                d, seconds = report.take()
                n = max(d["frames"], 1)
                ms, latency = d["busy"] / n * 1e3, d["latency"] / n * 1e3
                values = {"frames/s": d["frames"] / seconds, "ms/frame": ms, "objects/frame": d["objects"] / n,
                          "ms from frame to results": latency}
                if overlay is not None:
                    values |= {"overlays published": overlay.published, "overlays replaced": overlay.replaced}
                waiting = None if d["frames"] else f"{color_base}: {color.describe()}"
                level, message = node.state(f"{d['frames'] / seconds:.2f} frames/s", [], waiting)
                report.publish(level, message, values, log_extra=f", {ms:.0f} ms a frame")
    except KeyboardInterrupt:
        pass
    finally:
        history.close()  # and its subscriber
        if overlay is not None:
            overlay.close()
            overlay.publisher.close()
        detections_pub.close()
        report.close()
        params.close()
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
