"""The node's weights: which files it reads, where they come from, and their download.

This module is the one list of them: the node's start and ``tools/setup.sh`` (through ``xblick-yolox
--download-only``) both use it. ``ensure`` touches neither the disk nor the network when every file is there. A
missing file is downloaded under a temporary name in its directory (``<name>.part``), checked and only then renamed,
all under a file lock, so that two nodes started at once download once and never read half a file.

* ``MODELS``: name -> (URL, size in bytes, SHA-256) of the ONNX file Megvii publishes in YOLOX's release 0.1.1rc0, in
  ``weights/yolox_<name>.onnx``; size and SHA-256 measured from the files downloaded on 2026-11-14 (the release states
  no checksums); the size equals the one GitHub's release lists
* ``needed(model)``: the file a start with ``--model`` reads

The rest of this file (from ``Weight`` on) is the same in every node's ``weights.py``.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROG = "xblick-yolox"
REQUIRED = ()  # (path, what): code from tools/setup.sh that must be there first; none here
REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "weights"
RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
MODELS = {  # name -> (URL, size, SHA-256): into weights/yolox_<name>.onnx
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


def path_of(model: str) -> Path:
    """Where the ONNX file of ``--model`` is."""
    return WEIGHTS / f"yolox_{model}.onnx"


def needed(model: str) -> list[Weight]:
    """The file of ``--model`` (a name of MODELS)."""
    url, size, sha256 = MODELS[model]
    return [Weight(path_of(model), Url(url), size, sha256)]


# ---------------------------------------------------------------------------- the same in every node's weights.py

CHUNK = 1 << 20  # bytes read at a time
TIMEOUT_S = 60.0  # without a byte for this long, a download fails
LOCK = ".download.lock"  # in WEIGHTS: one lock for every download of the node


class WeightsError(Exception):
    """Why the weights are not there: one line for the user, and the exit code (130 Ctrl-C, 143 SIGTERM)."""

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


class _Terminated(BaseException):
    """SIGTERM during a download (a BaseException: no ``except Exception`` on the way swallows it)."""


@dataclass(frozen=True)
class HuggingFace:
    """A file of a Hugging Face repository at a fixed revision; ``gated``: the repository wants an accepted licence
    and a login."""

    repo: str
    revision: str
    filename: str
    gated: bool = False

    def describe(self) -> str:
        return f"Hugging Face {self.repo} at revision {self.revision}"

    def url(self) -> str:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        return f"{endpoint}/{self.repo}/resolve/{self.revision}/{urllib.parse.quote(self.filename)}"

    def fetch(self, tmp: Path, progress) -> str:
        if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on"):
            raise WeightsError(f"HF_HUB_OFFLINE is set, so {self.filename} is not downloaded from {self.repo}: unset "
                               "it, or copy the file from a PC that has it")
        denied = None
        if self.gated:
            denied = (f"{self.repo} is gated: accept its licence at https://huggingface.co/{self.repo} with your "
                      "Hugging Face account, then log in once with .venv/bin/hf auth login or set HF_TOKEN")
        return _http(self.url(), tmp, progress, token=_hf_token(), denied=denied)


@dataclass(frozen=True)
class Url:
    """A file at a URL."""

    url: str

    def describe(self) -> str:
        return self.url

    def fetch(self, tmp: Path, progress) -> str:
        return _http(self.url, tmp, progress)


@dataclass(frozen=True)
class Weight:
    """One file the node reads: where it is (``path``), where it comes from (``source``), its exact size (``size``,
    bytes: announced, and a download of another length is refused) and, where known, its SHA-256."""

    path: Path
    source: object  # HuggingFace, Url or a source of the node's own: describe() and fetch(tmp, progress) -> SHA-256
    size: int
    sha256: str | None = None


def add_arguments(parser):
    """``--download-only`` and ``--no-download``, which exclude each other."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download-only", action="store_true",
                       help="download the missing weights, check them and end (for preparation scripts): no camera, "
                            "no Zenoh, no GPU; the options a start requires may be left out")
    group.add_argument("--no-download", action="store_true",
                       help="never download: end at once when weights are missing (for machines that must not start "
                            "a download on their own)")


def parse_args(parser, argv=None):
    """``parser.parse_args(argv)``. With ``--download-only`` (spelled out) the options a start requires (``--camera``
    ...) are not required: a preparation script names only what selects the weights."""
    tokens = list(sys.argv[1:] if argv is None else argv)
    options = tokens[:tokens.index("--")] if "--" in tokens else tokens
    if "--download-only" not in options:
        return parser.parse_args(argv)
    required = [action for action in parser._actions if action.required]  # argparse keeps no public list
    for action in required:
        action.required = False
    try:
        return parser.parse_args(argv)
    finally:
        for action in required:
            action.required = True


def download_only(weights: list[Weight], out=None) -> int:
    """``--download-only``: the missing weights downloaded, every one with a known SHA-256 checked (also those there
    already), one line at the end; the exit code."""
    out = out or sys.stdout
    try:
        got = ensure(weights, verify=True, out=out)
    except WeightsError as error:
        return fail(error)
    if not weights:
        print(f"{PROG}: no weights to download for these options", file=out, flush=True)
    elif got:
        print(f"{PROG}: weights complete: {len(got)} downloaded into {_short(_root(weights))}", file=out, flush=True)
    else:
        print(f"{PROG}: weights complete in {_short(_root(weights))}, nothing to download", file=out, flush=True)
    return 0


def fail(error: WeightsError) -> int:
    """The error's line on stderr; its exit code."""
    print(f"{PROG}: {error}", file=sys.stderr, flush=True)
    return error.code


def ensure(weights: list[Weight], *, download: bool = True, verify: bool = False, out=None) -> list[Weight]:
    """Every file of ``weights`` there, or WeightsError (also when a path of REQUIRED is missing: code that comes with
    tools/setup.sh). All there: returns [] at once, no network, no file written
    (``verify``: those with a SHA-256 are read and checked). Missing ones (``download`` False: WeightsError naming
    them): under the lock, the model's terms and what comes, then each file downloaded, checked and renamed. Returns
    those downloaded."""
    out = out or sys.stdout
    for path, what in REQUIRED:
        if not path.exists():
            raise WeightsError(f"{_short(path)} is missing ({what}): run tools/setup.sh")
    missing = [w for w in weights if not w.path.is_file()]
    with _stoppable():
        for w in weights if verify else []:
            if w not in missing and w.sha256 is not None:
                digest = _sha256(w.path)
                if digest != w.sha256:
                    raise WeightsError(f"{_short(w.path)}: wrong SHA-256 ({digest}, expected {w.sha256}): remove "
                                       "it, then download again")
    if not missing:
        return []
    names = ", ".join(_short(w.path) for w in missing)
    if not download:
        raise WeightsError(f"weights missing: {names}; download them with {PROG} --download-only or tools/setup.sh "
                           "(--no-download is set)")
    with _stoppable(), _locked(WEIGHTS / LOCK, out):
        missing = [w for w in missing if not w.path.is_file()]  # another process may have downloaded them meanwhile
        if not missing:
            print(f"{PROG}: another process downloaded the weights", file=out, flush=True)
            return []
        total = sum(w.size for w in missing)
        print(f"{PROG}: to download: " + ", ".join(f"{_short(w.path)} ({_size(w.size)})" for w in missing)
              + f", {_size(total)} in all", file=out)
        print(TERMS, file=out, flush=True)
        for w in missing:
            _download(w, out)
    return missing


def _download(w: Weight, out):
    """One file: into ``<name>.part``, its size and (where known) its SHA-256 checked, renamed. A failed check keeps
    the .part file for a look; any other failure or an interrupt removes it."""
    tmp = w.path.with_name(w.path.name + ".part")
    w.path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{PROG}: {w.path.name} ({_size(w.size)}) from {w.source.describe()}", file=out, flush=True)
    progress = _Progress(w.path.name, out)
    try:
        digest = w.source.fetch(tmp, progress)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        progress.end()
    if (length := tmp.stat().st_size) != w.size:
        raise WeightsError(f"{_short(tmp)}: {length} bytes, expected {w.size}: the source changed or the download is "
                           "damaged; remove it, then start again")
    if w.sha256 is not None and digest != w.sha256:
        raise WeightsError(f"{_short(tmp)}: wrong SHA-256 ({digest}, expected {w.sha256}): the source changed or the "
                           "download is damaged; remove it, then start again")
    os.replace(tmp, w.path)
    print(f"{PROG}: {_short(w.path)}: " + ("size and SHA-256 checked" if w.sha256 else "size checked") + ", in place",
          file=out, flush=True)


def _http(url: str, tmp: Path, progress, token: str | None = None, denied: str | None = None) -> str:
    """GET ``url`` into ``tmp`` (redirects followed; ``token`` as a bearer to ``url``'s host only); the SHA-256 of
    what came. ``denied``: the message for HTTP 401 and 403."""
    request = urllib.request.Request(url, headers={"User-Agent": PROG})
    if token:
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    host = urllib.parse.urlsplit(url).netloc
    digest, done = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response, open(tmp, "wb") as f:
            length = response.headers.get("Content-Length")
            total = int(length) if length is not None else None
            progress(0, total)
            while chunk := response.read(CHUNK):
                f.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                progress(done, total)
            f.flush()
            os.fsync(f.fileno())
    except urllib.error.HTTPError as error:
        if error.code in (401, 403) and denied:
            raise WeightsError(denied) from None
        raise WeightsError(f"{url}: HTTP {error.code} {error.reason}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        reason = getattr(error, "reason", error)
        raise WeightsError(f"cannot download from {host} ({reason}): connect this PC to the internet, or run "
                           f"{PROG} --download-only on one that is and copy {_short(WEIGHTS)}") from None
    except OSError as error:
        raise WeightsError(f"{_short(tmp)}: {error.strerror or error}") from None
    if total is not None and done != total:
        raise WeightsError(f"{url}: {done} of {total} bytes came: start again")
    return digest.hexdigest()


def _hf_token() -> str | None:
    """The Hugging Face token: huggingface_hub's (HF_TOKEN, else the login's), else HF_TOKEN."""
    try:
        from huggingface_hub import get_token  # noqa: PLC0415 - only where a download needs it
    except ImportError:
        return os.environ.get("HF_TOKEN") or None
    return get_token()


class _Progress:
    """Progress of one file: on a terminal a line that updates in place, else (a log file) a line per tenth."""

    def __init__(self, name: str, out):
        self.name, self.out = name, out
        self.tty = bool(getattr(out, "isatty", lambda: False)())
        self.start = self.shown = time.monotonic()
        self.step = -1  # the last tenth printed
        self.drawn = False

    def __call__(self, done: int, total: int | None):
        now = time.monotonic()
        rate = done / max(now - self.start, 1e-6) / 1e6
        text = f"{_size(done)} of {_size(total)}" if total else _size(done)
        if self.tty:
            if now - self.shown >= 0.25 or (total and done == total):
                percent = f"{100 * done // total:3d}% " if total else ""
                self.out.write(f"\r{PROG}: {self.name}: {percent}{text}, {rate:.1f} MB/s   ")
                self.out.flush()
                self.shown, self.drawn = now, True
            return
        step = 10 * done // total if total else done // (500 * 10**6)  # without a length: every 500 MB
        if step > self.step and done:
            self.step = step
            percent = f"{100 * done // total}% " if total else ""
            print(f"{PROG}: {self.name}: {percent}({text}, {rate:.1f} MB/s)", file=self.out, flush=True)

    def end(self):
        if self.drawn:
            self.out.write("\n")
            self.out.flush()


@contextlib.contextmanager
def _locked(path: Path, out):
    """The lock of this node's downloads (fcntl): another process downloading, wait for it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"{PROG}: another process downloads the weights; waiting for it", file=out, flush=True)
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@contextlib.contextmanager
def _stoppable():
    """Ctrl-C and SIGTERM end a download or a check at once: WeightsError, 130 or 143 (the .part file removed)."""
    main = threading.current_thread() is threading.main_thread()

    def terminate(signum, frame):
        raise _Terminated

    previous = signal.signal(signal.SIGTERM, terminate) if main else None
    try:
        yield
    except KeyboardInterrupt:
        raise WeightsError("stopped (Ctrl-C); no partial file kept", 130) from None
    except _Terminated:
        raise WeightsError("stopped (SIGTERM); no partial file kept", 143) from None
    finally:
        if main:
            signal.signal(signal.SIGTERM, previous)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _root(weights: list[Weight]) -> Path:
    return Path(os.path.commonpath([w.path.parent for w in weights]))


def _short(path: Path) -> str:
    """``path`` relative to the repository where it is in it."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _size(n: int | None) -> str:
    if n is None:
        return "?"
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"
