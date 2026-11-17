"""``--ros`` (the core's branch ros2): the node with the stand-in model as its own process. With ``--ros`` its
detections and its overlay also appear on rmw_zenoh's keys, with ROS's 33-byte attachment and liveliness tokens, and a
ROS subscriber reads them; without it nothing of ROS goes on the wire."""

import signal
import subprocess
import sys
import threading
import time

import numpy as np
import xblick as db
from PIL import Image
from test_node import CAM, OFFICE, STAND_IN, free_port
from xblick import ros, topics
from xblick.session import RosOptions, open_session

DOMAIN = 93
DETECTIONS = f"/{topics.detections(CAM, 'yolox', 'detections_2d')}"
OVERLAY = f"/{CAM}/yolox/overlay/image_raw"


def test_every_default_key_is_a_ros_name():
    for key in (f"{CAM}/color", topics.detections(CAM, "yolox", "detections_2d"), f"{CAM}/yolox/overlay/image_raw",
                f"{CAM}/yolox/overlay/camera_info", f"{CAM}/diagnostics"):
        ros.topic_of(key)


def run_node(with_ros: bool, seconds: float = 30.0):
    """The stand-in node with ``--overlay`` (and ``--ros``) on a published office frame. A reader that takes part in
    ROS subscribes to the ROS topics of the detections and the overlay (a mirror sends only to a ROS subscriber); an
    observer that takes part in nothing records every key with its attachment and every ROS liveliness token.
    Returns ({ROS topic: [payload]}, {key: attachment}, {token}, the node's output)."""
    bgr = np.ascontiguousarray(np.asarray(Image.open(OFFICE).convert("RGB"))[..., ::-1])
    port = free_port()
    endpoint = f"tcp/127.0.0.1:{port}"
    session = open_session(listen=[endpoint], ros_options=RosOptions(DOMAIN, "xblick_reader") if with_ros else None)
    observer = open_session(connect=[endpoint], declare=False)
    keys: dict[str, bytes | None] = {}
    tokens: set[str] = set()
    got: dict[str, list[bytes]] = {DETECTIONS: [], OVERLAY: []}
    watch = observer.declare_subscriber(
        "**", lambda s: keys.setdefault(str(s.key_expr), s.attachment.to_bytes() if s.attachment else None))
    live = observer.liveliness().declare_subscriber("@ros2_lv/**", lambda s: tokens.add(str(s.key_expr)),
                                                    history=True)
    readers = []
    if with_ros:
        readers = [session.ros.node.subscriber(DETECTIONS, db.Detection2DArray.TYPE,
                                               lambda payload, *_: got[DETECTIONS].append(payload)),
                   session.ros.node.subscriber(OVERLAY, "sensor_msgs/msg/Image",
                                               lambda payload, *_: got[OVERLAY].append(payload))]
    publisher = db.CameraPublisher(session, f"{CAM}/color", encoding="bgr8", frame_id="office", source="test")
    # --ros-node: the stand-in is python -c, whose program name is not xblick-yolox
    options = ["--overlay"] + (["--ros", "--ros-domain", str(DOMAIN), "--ros-node", "xblick_yolox"] if with_ros else [])
    process = subprocess.Popen([sys.executable, "-c", STAND_IN, "--camera", CAM, "--connect", endpoint, *options],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output: list[str] = []
    threading.Thread(target=lambda: output.extend(process.stdout), daemon=True).start()
    try:
        stamp_ns, deadline = 1_000_000_000, time.monotonic() + seconds
        overlay_seen = 0
        while time.monotonic() < deadline:
            assert process.poll() is None, "".join(output)
            publisher.publish(bgr, stamp_ns=stamp_ns)
            stamp_ns += 100_000_000
            time.sleep(0.1)
            overlay_seen += any(k.endswith("yolox/overlay/image_raw/compressed") for k in list(keys))
            if (got[DETECTIONS] and got[OVERLAY]) if with_ros else overlay_seen >= 10:
                break
        time.sleep(0.3)
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=30)
        for reader in readers:
            reader.close()
        publisher.close()
        watch.undeclare()
        live.undeclare()
        observer.close()
        session.close()
    assert process.returncode == 0, "".join(output)
    return got, keys, tokens, "".join(output)


def test_with_ros_the_detections_and_the_overlay_reach_ros():
    got, keys, tokens, output = run_node(with_ros=True)
    (message, *_) = [db.Detection2DArray.deserialize(p) for p in got[DETECTIONS]]
    assert [d.results[0].hypothesis.class_id for d in message.detections] == ["person", "cup"]
    assert message.header.frame_id == "office" and got[OVERLAY]
    for topic, type_name in ((DETECTIONS, db.Detection2DArray.TYPE), (OVERLAY, "sensor_msgs/msg/Image")):
        key = ros.data_key(DOMAIN, topic, type_name, ros.hash_of(type_name))
        assert len(keys[key]) == 33, key
        seq, stamp, _ = ros.decode_attachment(keys[key])
        assert seq >= 1 and abs(stamp - time.time_ns()) < 60 * 10**9
    parsed = [ros.parse_token(t) for t in tokens]
    publishers = {t.topic for t in parsed if t.kind == ros.PUBLISHER and t.fully_qualified_node == "/xblick_yolox"}
    assert {DETECTIONS, OVERLAY} <= publishers, publishers
    assert {"/test/camera/yolox", "/xblick_yolox"} <= {t.fully_qualified_node for t in parsed if t.kind == ros.NODE}
    assert topics.detections(CAM, "yolox", "detections_2d") in keys  # our keys as before


def test_without_ros_nothing_of_ros_goes_on_the_wire():
    _, keys, tokens, _ = run_node(with_ros=False)
    assert topics.detections(CAM, "yolox", "detections_2d") in keys
    assert not [k for k in keys if "::dds_::" in k or k[:1].isdigit()] and tokens == set()
