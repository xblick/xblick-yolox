"""The node's overlay drawing (through xblick.overlay.draw_boxes): image in, image out, the boxes where the
detections are."""

import numpy as np
from xblick.overlay import colour

from xblick_yolox.model import Detections
from xblick_yolox.node import draw


def test_boxes_are_drawn_in_their_class_colour_where_the_detections_are():
    rgb = np.zeros((480, 640, 3), np.uint8)
    found = Detections(np.array([[100, 200, 300, 400], [400, 50, 500, 150]], np.float32),
                       np.array([0.9, 0.5], np.float32), np.array([0, 41]))
    out = draw(rgb, found)
    assert out.shape == rgb.shape and out.dtype == np.uint8 and not rgb.any()  # a copy, the input unchanged
    assert tuple(out[300, 100]) == colour(0) and tuple(out[300, 300]) == colour(0)  # left and right edge
    assert tuple(out[399, 200]) == colour(0) and tuple(out[100, 400]) == colour(41)
    assert not out[300, 200].any() and not out[100, 450].any()  # inside the boxes: the frame as it was
    assert not out[450:, :50].any()  # far from any box: nothing drawn
    label = out[175:198, 100:180]  # the label above the person's box: its colour and black text
    assert (label == colour(0)).all(-1).any() and (label == 0).all(-1).any()


def test_no_detections_give_the_frame_as_it_was():
    rgb = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    assert (draw(rgb, Detections.empty()) == rgb).all()
