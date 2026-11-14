# How the node works

This page describes the model's input and output, the post-processing, the overlay and the tests.

## The model

The node runs the ONNX files of YOLOX's release 0.1.1rc0 in ONNX Runtime. Each file has a fixed input size: 416 x 416
for nano and tiny, 640 x 640 for s. The node checks the input and output shapes when it loads a file.

The input `images` is a (1, 3, S, S) float32 array. The node scales the frame by one factor so that it fits into
S x S, with Pillow's bilinear scaling. It puts the frame into the top-left corner and fills the rest with the grey
value 114. The channels are in BGR order with values from 0 to 255, because YOLOX trained on such images.

The output `output` is an (N, 85) array, one row per anchor point. The anchor points are the cells of three grids
with strides 8, 16 and 32 over the input, each grid row by row. A row holds x, y, w and h, the objectness and 80 class
scores. The node decodes the centre as `(x + cell) * stride` and the size as `exp(w) * stride`.

## The post-processing

1. The node takes the best class of each anchor point. Its score is the objectness times that class's score.
2. It keeps the anchor points with a score of at least `--score`, and only the classes of `--classes`.
3. It suppresses boxes per class (greedy non-maximum suppression with `--nms`). A box never suppresses a box of
   another class.
4. It divides the boxes by the scale factor and clips them to the frame.

`model.py` holds all of it in numpy. It follows the documented format and copies no code of YOLOX.

## The overlay

The overlay convention, which later nodes follow:

| Rule | This node |
| --- | --- |
| Option | `--overlay`, off by default, read-only (it applies at the start) |
| Key | `<camera>/<node>/overlay`, an ordinary image stream |
| Encoding | `rgb8`, the colour frame's size, its stamp, its camera frame and its camera_info. Remote readers get JPEG. |
| Without the option | no publisher, so no topic and no info record |
| Order | the node publishes the detections first, then hands the frame to the overlay thread |
| Load | the thread draws only while a subscriber exists. It keeps one frame, and a newer frame replaces a waiting one. |
| camera_info | the colour stream's, forwarded with each overlay frame |
| Code | the core's `xblick.overlay`: `Overlay` publishes and draws in its thread, `draw_boxes` draws. The node's `draw` gives the texts and colours. |

The drawing shows each box in a fixed colour per class, with the class name and the score above the box. The best
detection is drawn last, so it lies on top. The diagnostics count the overlays published and replaced.

## Tests

| File | What |
| --- | --- |
| `tests/test_model.py` | letterbox, grids, decoding and suppression on synthetic outputs, the real models on `office.png` (skip without `weights/`) |
| `tests/test_overlay.py` | the drawing: boxes and labels where the detections are |
| `tests/test_node.py` | conversions and options, the node as its own process with a stand-in model and with the real tiny model |
| `tests/test_weights.py` | the model list, `--download-only`, `--no-download` and a start without the model, against a local HTTP server |

The stand-in model (`tests/standin.py`) replaces only ONNX Runtime. It returns an output in which a person and a cup
are found, so the node's pre- and post-processing run as they do with the real model.
