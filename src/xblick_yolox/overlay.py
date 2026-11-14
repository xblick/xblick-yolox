"""The overlay: a colour frame with the detections drawn on it (``draw``), and the thread that draws and publishes it
after the detections are out (``OverlayWorker``).

    worker = OverlayWorker(publisher, names)
    worker.put(stamp_ns, frame_id, rgb, detections)    # after detections_pub.publish(...): never waits
    worker.check()                                     # raises the worker's error here
"""

from __future__ import annotations

import colorsys
import threading

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .model import Detections


def colour(label: int) -> tuple[int, int, int]:
    """A fixed colour per class: hues spread by the golden ratio, so that neighbouring indices differ."""
    r, g, b = colorsys.hsv_to_rgb((label * 0.618033988749895) % 1.0, 0.85, 1.0)
    return round(r * 255), round(g * 255), round(b * 255)


def draw(rgb: np.ndarray, found: Detections, names: tuple[str, ...]) -> np.ndarray:
    """A copy of the (H, W, 3) uint8 RGB frame with each detection's box, class name and score
    (``person 0.87``) drawn in its class's colour; the best detection last, so on top."""
    image = Image.fromarray(rgb)
    pen = ImageDraw.Draw(image)
    h, w = rgb.shape[:2]
    width = max(2, round(min(w, h) / 240))  # 2 pixels at 480 lines
    font = ImageFont.load_default(size=max(12, round(min(w, h) / 32)))
    for i in reversed(range(len(found))):
        x1, y1, x2, y2 = (float(v) for v in found.boxes[i])
        fill = colour(int(found.labels[i]))
        pen.rectangle((x1, y1, x2, y2), outline=fill, width=width)
        text = f"{names[int(found.labels[i])]} {float(found.scores[i]):.2f}"
        left, top, right, bottom = pen.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top
        ty = y1 - th - 2 * width if y1 - th - 2 * width >= 0 else y1  # above the box, else inside it
        pen.rectangle((x1, ty, x1 + tw + 2 * width, ty + th + 2 * width), fill=fill)
        pen.text((x1 + width - left, ty + width - top), text, fill=(0, 0, 0), font=font)
    return np.asarray(image)


class OverlayWorker:
    """Draws and publishes the overlay in a thread of its own, so the detections never wait for it. It keeps one
    frame: a frame put while it still draws the last replaces the one waiting (counted in ``replaced``). It draws only
    while somebody subscribes to the overlay (``publisher.subscribed``)."""

    def __init__(self, publisher, names: tuple[str, ...]):
        self.publisher = publisher
        self.names = names
        self.published = 0
        self.replaced = 0
        self._waiting = None
        self._error: BaseException | None = None
        self._stop = False
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="overlay", daemon=True)
        self._thread.start()

    def put(self, stamp_ns: int, frame_id: str, rgb: np.ndarray, found: Detections):
        with self._cond:
            if self._waiting is not None:
                self.replaced += 1
            self._waiting = (stamp_ns, frame_id, rgb, found)
            self._cond.notify()

    def _run(self):
        try:
            while True:
                with self._cond:
                    while self._waiting is None and not self._stop:
                        self._cond.wait()
                    if self._stop:
                        return
                    (stamp_ns, frame_id, rgb, found), self._waiting = self._waiting, None
                if not self.publisher.subscribed:
                    continue
                image = draw(rgb, found, self.names)
                self.publisher.frame_id = frame_id
                self.publisher.publish(image, stamp_ns=stamp_ns)
                self.published += 1
        except BaseException as error:  # noqa: BLE001 - handed to the node's loop by check()
            self._error = error

    def check(self):
        """Raises the error that ended the worker, in the caller's thread."""
        if self._error is not None:
            raise RuntimeError(f"the overlay failed: {type(self._error).__name__}: {self._error}") from self._error

    def close(self):
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout=5.0)
