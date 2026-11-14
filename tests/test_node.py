"""The node: its conversions and options, and the node itself as its own process on a published colour stream over
real Zenoh sessions, with a stand-in model (always) and with the real tiny model (when the node's cache directory
has it)."""

import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import xblick as db
from PIL import Image
from xblick import topics
from xblick.camera import query_info
from xblick.session import open_session

from xblick_yolox import node, weights
from xblick_yolox.model import Detections

CAM = "test/camera"
OFFICE = Path(__file__).parent / "data" / "office.png"


def test_colour_frames_become_rgb():
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    assert (node.to_rgb(rgb, "rgb8") == rgb).all()
    assert (node.to_rgb(np.ascontiguousarray(rgb[..., ::-1]), "bgr8") == rgb).all()
    assert (node.to_rgb(np.dstack([rgb, np.zeros((2, 3), np.uint8)]), "rgba8") == rgb).all()
    assert (node.to_rgb(rgb[..., 0], "mono8") == rgb[..., :1]).all()
    with pytest.raises(ValueError, match="16UC1"):
        node.to_rgb(rgb[..., 0], "16UC1")


def test_classes_are_coco_names_also_comma_separated():
    assert node.split_classes(["person,cup", "cell phone"]) == ["person", "cup", "cell phone"]
    assert node.class_indices(["person", "cup"]) == {0, 41} and node.class_indices([]) is None
    with pytest.raises(ValueError, match="no COCO class: mug"):
        node.class_indices(["mug"])
    assert node.check_classes(["person"]) is None and "mug" in node.check_classes(["mug"])


def test_options_and_their_fixed_defaults():
    args = node.make_parser().parse_args(["--camera", CAM])
    assert (args.model, args.score, args.nms, args.classes, args.overlay, args.threads, args.device, args.max_hz,
            args.node) == ("tiny", 0.3, 0.45, [], False, 4, "cpu", None, "yolox")
    assert node.parse_device("cuda") == "cuda:0" and node.parse_device("cuda:2") == "cuda:2"
    for bad in (["--model", "medium"], ["--model", "darknet"], ["--device", "gpu"], ["--local", "cuda"]):
        with pytest.raises(SystemExit):
            node.make_parser().parse_args(["--camera", CAM, *bad])


def test_the_start_line_says_where_the_model_computes():
    assert node.where_it_computes("cpu", 4) == "model on the CPU with 4 threads"
    assert node.where_it_computes("cuda:0", 4) == "model on cuda:0; scaling, decoding and drawing on the CPU"


def test_cuda_without_the_gpu_runtime_names_the_one_step(tmp_path, monkeypatch):
    from xblick.weights import Store

    from xblick_yolox import gpu

    monkeypatch.setattr(weights, "STORE", Store(weights.PROG, root=tmp_path, terms="", repo=tmp_path))
    with pytest.raises(RuntimeError) as error:
        gpu.onnxruntime_for("cuda:0")
    message = str(error.value)
    assert "tools/setup.sh --gpu" in message and str(tmp_path / "onnxruntime-gpu") in message
    assert "\n" not in message and "CUDA 13" in message


def test_a_missing_camera_option_ends_with_one_line(capsys):
    with pytest.raises(SystemExit) as exit_:
        node.main([])
    assert exit_.value.code == 2 and "--camera" in capsys.readouterr().err


def test_detections_message_carries_names_scores_and_boxes():
    found = Detections(np.array([[10, 20, 30, 60]], np.float32), np.array([0.75], np.float32), np.array([41]))
    header = db.Header(db.Time.from_ns(5_000_000_123), "cam")
    message = node.to_message(found, header)
    (d,) = message.detections
    assert message.header == header and d.results[0].hypothesis.class_id == "cup"
    assert d.results[0].hypothesis.score == pytest.approx(0.75)
    assert (d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.size_x, d.bbox.size_y) == (20, 40, 20, 40)
    assert db.Detection2DArray.deserialize(message.serialize()) == message


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# the node with the stand-in model (standin.py) and no weights: the rest is the node as it is
STAND_IN = f"""
import sys
sys.path.insert(0, {str(Path(__file__).parent)!r})
from standin import StandIn
from xblick_yolox import node, weights
weights.needed = lambda model: []
node.load_detector = lambda path, threads, device: StandIn(path, threads, device)
sys.exit(node.main(sys.argv[1:]))
"""


def run_node(options: list[str], stand_in: bool, frames: int = 5, timeout: float = 60.0):
    """Publishes the office frame (bgr8) at 10 Hz with its own stamps; the node, its own process, answers. Returns
    {stamp: (receive time, Detection2DArray)}, {stamp: (receive time, overlay frame)}, the overlay's info records
    and the node's output."""
    bgr = np.ascontiguousarray(np.asarray(Image.open(OFFICE).convert("RGB"))[..., ::-1])
    port = free_port()
    session = open_session(listen=[f"127.0.0.1:{port}"])
    publisher = db.CameraPublisher(session, f"{CAM}/color", encoding="bgr8", frame_id="office", source="test")
    detections, overlays, lock = {}, {}, threading.Lock()

    def on_detections(message):
        with lock:
            detections[message.header.stamp.to_ns()] = (time.monotonic(), message)

    det_sub = db.DetectionSubscriber(session, topics.detections(CAM, "yolox", "detections_2d"), on_detections)
    overlay_sub = db.ImageSubscriber(session, f"{CAM}/yolox/overlay", transport="compressed")
    command = ([sys.executable, "-c", STAND_IN] if stand_in
               else [shutil.which("xblick-yolox", path=os.path.dirname(sys.executable))])
    process = subprocess.Popen([*command, "--camera", CAM, "--connect", f"127.0.0.1:{port}", *options],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output: list[str] = []
    threading.Thread(target=lambda: output.extend(process.stdout), daemon=True).start()
    want_overlay = "--overlay" in options
    try:
        stamp_ns, deadline = 1_000_000_000, time.monotonic() + timeout
        while time.monotonic() < deadline:
            assert process.poll() is None, "".join(output)
            publisher.publish(bgr, stamp_ns=stamp_ns)
            stamp_ns += 100_000_000
            end = time.monotonic() + 0.1
            while time.monotonic() < end:
                frame = overlay_sub.latest(timeout=0.02)
                if frame is not None:
                    overlays[frame.header.stamp.to_ns()] = (time.monotonic(), frame)
            det_sub.check()
            with lock:
                done = len(detections) >= frames and (not want_overlay or len(overlays) >= frames)
            if done:
                break
        records = query_info(session, f"{CAM}/yolox/overlay")
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=30)
        overlay_sub.close()
        det_sub.close()
        publisher.close()
        session.close()
    assert process.returncode == 0, "".join(output)
    return detections, overlays, records, "".join(output)


def check_detections(detections, names_wanted):
    assert len(detections) >= 5
    for _, message in detections.values():
        assert message.header.frame_id == "office"
        names = [d.results[0].hypothesis.class_id for d in message.detections]
        scores = [d.results[0].hypothesis.score for d in message.detections]
        assert scores == sorted(scores, reverse=True)
        names_wanted(names)


def test_node_with_a_stand_in_publishes_detections_and_no_overlay_by_default():
    detections, overlays, records, output = run_node([], stand_in=True)
    check_detections(detections, lambda names: names == ["person", "cup"])
    assert overlays == {} and records == []  # no overlay stream at all: no publisher, no info record
    assert "-> test/camera/yolox/{detections_2d}" in output


def test_node_with_a_stand_in_and_overlay_publishes_the_drawn_frame_after_the_detections():
    detections, overlays, records, _ = run_node(["--overlay"], stand_in=True)
    check_detections(detections, lambda names: names == ["person", "cup"])
    assert len(records) == 1 and records[0]["encoding"] == "rgb8" and "compressed" in records[0]["transports"]
    assert len(overlays) >= 5
    for stamp, (received, frame) in overlays.items():
        assert stamp in detections and detections[stamp][0] <= received  # the results first
        image = np.asarray(frame.image)
        assert image.shape == (480, 640, 3) and frame.header.frame_id == "office"


def test_node_restricted_to_a_class_publishes_only_that_class():
    detections, _, _, _ = run_node(["--classes", "cup", "--score", "0.5"], stand_in=True)
    check_detections(detections, lambda names: names == ["cup"])


@pytest.mark.skipif(not weights.path_of("tiny").is_file(), reason="no yolox_tiny.onnx in the node's cache directory")
def test_node_with_the_real_tiny_model_finds_chairs_and_draws_them():
    detections, overlays, _, _ = run_node(["--overlay", "--no-download"], stand_in=False)
    check_detections(detections, lambda names: names.count("chair") >= 2 and "bottle" in names)
    assert len(overlays) >= 5
