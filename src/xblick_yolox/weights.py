"""The node's weights: which files it reads and where they come from. ``xblick.weights`` downloads and checks them.

This module is the one list of them: the node's start and ``tools/setup.sh`` (through ``xblick-yolox
--download-only``) both use it.

* ``MODELS``: name -> (URL, size in bytes, SHA-256) of the ONNX file Megvii publishes in YOLOX's release 0.1.1rc0, in
  ``yolox_<name>.onnx`` of the node's cache directory (``~/.cache/xblick/yolox``, ``--weights-dir``); size and
  SHA-256 measured from the files downloaded on 2026-11-14 (the release states no checksums); the size equals the one
  GitHub's release lists
* ``needed(model)``: the file a start with ``--model`` reads, its path relative to the store's root; ``path_of``:
  where it is in a store
* ``store_for(parser, args)``: ``STORE`` at ``--weights-dir``; ``options(args)``: the options that select the weights,
  for the command the lines name
* ``STORE``: the download (``xblick.weights.Store``), its root the node's cache directory. The checkout's ``weights/``
  is the node's former place: a model still there is not used, and the store prints the command that moves it.
"""

from __future__ import annotations

from pathlib import Path

from xblick import weights

PROG = "xblick-yolox"
NODE = "yolox"  # the short name: its cache directory is ~/.cache/xblick/yolox
REQUIRED = ()  # (path, what): code from tools/setup.sh that must be there first; none here
REPO = Path(__file__).resolve().parents[2]
FORMER = REPO / "weights"  # the models' place before 0.1.2 (the checkout): not used, the store names the move
RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
MODELS = {  # name -> (URL, size, SHA-256): into <root>/yolox_<name>.onnx
    "nano": (f"{RELEASE}/yolox_nano.onnx", 3_659_407,
             "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"),
    "tiny": (f"{RELEASE}/yolox_tiny.onnx", 20_219_662,
             "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"),
    "s": (f"{RELEASE}/yolox_s.onnx", 35_858_002,
          "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"),
}
TERMS = """\
YOLOX is the work of Megvii, under its terms, and you are the one who uses it:
  * YOLOX's repository (github.com/Megvii-BaseDetection/YOLOX) is under the Apache License 2.0, as its LICENSE
    states. Its release 0.1.1rc0, the source of the model file, states no other terms for the trained models."""

# the download (xblick.weights): the root is the node's cache directory, its lock there
STORE = weights.Store(PROG, node=NODE, terms=TERMS, required=REQUIRED, repo=REPO, former=[FORMER])


def needed(model: str) -> list[weights.Weight]:
    """The file of ``--model`` (a name of MODELS), below the store's root."""
    url, size, sha256 = MODELS[model]
    return [weights.Weight(file_name(model), weights.Url(url), size, sha256)]


def file_name(model: str) -> Path:
    """The ONNX file of ``--model``, relative to a store's root."""
    return Path(f"yolox_{model}.onnx")


def path_of(model: str, store: weights.Store | None = None) -> Path:
    """Where the ONNX file of ``--model`` is in ``store`` (default STORE: the node's cache directory)."""
    return (store or STORE).root / file_name(model)


def options(args) -> list[str]:
    """The options that select the weights (and --weights-dir), for the download command the lines name."""
    return ["--model", args.model] + (["--weights-dir", args.weights_dir] if args.weights_dir else [])


def store_for(parser, args) -> weights.Store:
    """STORE at --weights-dir (not given: the node's cache directory); a wrong --weights-dir is an argument error."""
    try:
        return STORE.at(args.weights_dir)
    except ValueError as error:
        parser.error(f"--weights-dir: {error}")
