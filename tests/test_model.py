"""Pre- and post-processing on synthetic data (no model), and the real models on a frame of an office (they skip
without the models in the node's cache directory)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from standin import OBJECTS, StandIn, raw_for

from xblick_yolox import model, weights


def test_letterbox_scales_into_the_top_left_corner_in_bgr_and_pads_grey():
    rgb = np.zeros((240, 320, 3), np.uint8)
    rgb[...] = (10, 20, 30)
    blob, r = model.letterbox(rgb, 416)
    assert blob.shape == (1, 3, 416, 416) and blob.dtype == np.float32 and r == pytest.approx(1.3)
    assert tuple(blob[0, :, 0, 0]) == (30, 20, 10)  # BGR, values 0 to 255
    assert tuple(blob[0, :, 311, 415]) == (30, 20, 10)  # 240 * 1.3 = 312 rows, 320 * 1.3 = 416 columns
    assert (blob[0, :, 312:, :] == model.PAD).all()


def test_grids_follow_the_output_order():
    cells, strides = model.grids(416)
    assert len(cells) == 52 * 52 + 26 * 26 + 13 * 13 == 3549
    assert tuple(cells[1]) == (1, 0) and tuple(cells[52]) == (0, 1)  # row by row, x first
    assert strides[0, 0] == 8 and strides[52 * 52, 0] == 16 and strides[-1, 0] == 32 and tuple(cells[-1]) == (12, 12)


def test_grids_of_the_standard_models_input():
    cells, strides = model.grids(640)
    assert len(cells) == 80 * 80 + 40 * 40 + 20 * 20 == 8400  # the output rows of s, m, l, x and darknet53
    assert strides[80 * 80, 0] == 16 and strides[-1, 0] == 32 and tuple(cells[-1]) == (19, 19)


@pytest.mark.parametrize("size", sorted(set(model.INPUT_SIZE.values())))
def test_each_output_layout_decodes_the_synthetic_objects(size):
    cells, strides = model.grids(size)
    found = model.postprocess(raw_for(OBJECTS, size), cells, strides, r=size / 832, frame_size=(832, 832), score=0.3,
                              iou=0.45)
    assert found.labels.tolist() == [0, 41]
    expected = np.array([[40, 60, 200, 380], [260, 200, 300, 250]]) * 832 / size  # input pixels to frame pixels
    np.testing.assert_allclose(found.boxes, expected, atol=1e-2)


def test_decode_and_postprocess_find_the_synthetic_objects_in_frame_pixels():
    cells, strides = model.grids(416)
    found = model.postprocess(raw_for(OBJECTS, 416), cells, strides, r=0.5, frame_size=(832, 832), score=0.3, iou=0.45)
    assert found.labels.tolist() == [0, 41]
    np.testing.assert_allclose(found.scores, [0.9, 0.6], rtol=1e-6)
    np.testing.assert_allclose(found.boxes, [[80, 120, 400, 760], [520, 400, 600, 500]], atol=1e-3)
    assert len(model.postprocess(raw_for(OBJECTS, 416), cells, strides, 0.5, (832, 832), 0.7, 0.45)) == 1
    only_cups = model.postprocess(raw_for(OBJECTS, 416), cells, strides, 0.5, (832, 832), 0.3, 0.45, {41})
    assert only_cups.labels.tolist() == [41]
    assert len(model.postprocess(np.zeros((3549, 85), np.float32), cells, strides, 1.0, (416, 416), 0.3, 0.45)) == 0


def test_boxes_are_clipped_to_the_frame():
    cells, strides = model.grids(416)
    found = model.postprocess(raw_for([((-20.0, 10.0, 100.0, 450.0), 2, 0.8)], 416), cells, strides, 1.0,
                              (416, 300), 0.3, 0.45)
    np.testing.assert_allclose(found.boxes, [[0, 10, 100, 300]], atol=1e-3)


def test_nms_drops_overlaps_of_one_class_only():
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [20, 20, 30, 30], [0, 0, 10, 10]], np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.95], np.float32)
    assert model.nms(boxes, scores, 0.45).tolist() == [3, 2]
    assert model.nms(boxes, scores, 0.9).tolist() == [3, 1, 2]  # box 0: IoU 1 with box 3, goes; box 1: 0.82, stays
    cells, strides = model.grids(416)
    two = [((100.0, 100.0, 200.0, 200.0), 0, 0.9)]
    raw = raw_for(two, 416)
    i = int(np.flatnonzero(raw[:, 4])[0])
    raw[i - 1] = raw[i]
    raw[i - 1, 0] += 1.0  # the same box from the neighbouring cell
    raw[i - 1, 5:] = 0
    raw[i - 1, 5 + 56] = 0.8
    found = model.postprocess(raw, cells, strides, 1.0, (416, 416), 0.3, 0.45)
    assert sorted(found.labels.tolist()) == [0, 56]  # same box, other class: both stay


def test_the_stand_in_detector_goes_through_the_whole_pipeline():
    detector = StandIn(Path("none"), 4)
    found = detector.detect(np.zeros((480, 640, 3), np.uint8), 0.3, 0.45)
    assert found.labels.tolist() == [0, 41]
    np.testing.assert_allclose(found.boxes[0], np.minimum(np.array([40, 60, 200, 380]) * 640 / 416, 480), atol=1e-2)


OFFICE = Path(__file__).parent / "data" / "office.png"
# what stands on and around the desk
PLAUSIBLE = {"chair", "bottle", "cup", "dining table", "book", "tv", "laptop", "keyboard", "bowl", "scissors"}


@pytest.mark.parametrize("name", list(weights.MODELS))
def test_the_real_model_finds_the_office_objects(name):
    path = weights.path_of(name)
    if not path.is_file():
        pytest.skip(f"no {path.name} (xblick-yolox --download-only --model {name})")
    rgb = np.asarray(Image.open(OFFICE).convert("RGB"))
    detector = model.Detector(path, 4)
    assert detector.size == model.INPUT_SIZE[name]  # read from the file; the table documents it
    found = detector.detect(rgb, 0.3, 0.45)
    names = [model.COCO_CLASSES[i] for i in found.labels]
    # the frame shows two chairs and bottles on a desk: robust to the variant, no exact boxes
    assert names.count("chair") >= 2 and "bottle" in names, names
    assert found.scores[names.index("chair")] > 0.8
    assert set(names) <= PLAUSIBLE, names
    chairs = found.boxes[[n == "chair" for n in names]]
    assert (chairs[:, 3] > 400).all()  # both chairs stand in the lower part of the 480 rows
