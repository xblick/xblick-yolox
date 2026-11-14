"""ONNX Runtime for the nodes: a session on the CPU or a GPU (``open_session``, ``trial``, ``prepare``), and the GPU
runtime, ``onnxruntime-gpu`` with NVIDIA's CUDA and cuDNN libraries, in one directory outside the venv that every
ONNX Runtime node shares.

The venv holds ONNX Runtime for the CPU (``onnxruntime``, as ``uv.lock`` states). The GPU package cannot sit beside it:
both install the Python package ``onnxruntime``, and ``uv run`` or ``uv sync`` brings the CPU package back over a
swapped one. ``tools/setup.sh`` (``python -m <package>.gpu setup``) therefore installs the GPU runtime once into
``runtime_dir()``, by ``detect``'s rule when an NVIDIA GPU is present, with ``--gpu`` always, with ``--no-gpu`` never:

    ~/.cache/xblick/onnxruntime/gpu-<onnxruntime-gpu version>-cp<Python version>/

``xblick.paths.cache_dir("onnxruntime")``: ``XBLICK_HOME`` and ``XDG_CACHE_HOME`` move it as every cache. The version
and the Python version are in the name, because the package is built for one Python: nodes that pin the same version
share it, others get their own. uv never touches it, so it stays across ``uv run``, ``uv sync`` and ``tools/setup.sh``.
A node imports ONNX Runtime from there only when ``--device cuda`` asks for it:

    session, printed = open_session(path, threads=4, device="cuda:0", prog="xblick-yolox")   # RuntimeError: one line
    trial(session, {"images": zeros}, "cuda:0", printed)     # a GPU that loads but cannot compute: one line
    detector = prepare(lambda: Detector(...), "cuda:0", "xblick-yolox")   # the start line, Ctrl-C and SIGTERM work

Requirement of the pinned ``onnxruntime-gpu`` 1.32.0: Python 3.11 to 3.14 (it has no wheel for 3.10), CUDA 13 and
cuDNN 9 (NVIDIA's wheels below: its extras ``cuda`` and ``cudnn``), an NVIDIA driver for CUDA 13.

This file is the same in every ONNX Runtime node (xblick-yolox, xblick-rtdetrv4); change it in all of them.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import io
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

from xblick import paths

ONNXRUNTIME_GPU = "1.32.0"  # the version the install puts there, and the one the nodes accept
# the libraries of onnxruntime-gpu 1.32.0's extras cuda and cudnn (its package metadata), installed beside it
CUDA_LIBRARIES = ("nvidia-cuda-nvrtc~=13.0", "nvidia-cuda-runtime~=13.0", "nvidia-cufft~=12.0", "nvidia-curand~=10.0",
                  "nvidia-cudnn-cu13~=9.0")
PYTHONS = ((3, 11), (3, 14))  # the Pythons onnxruntime-gpu 1.32.0 has wheels for, first and last
SETUP = "tools/setup.sh --gpu"  # the one step, in a node's checkout
LOCK = ".install.lock"  # in cache_dir("onnxruntime"): one install at a time


def runtime_dir() -> Path:
    """Where the GPU runtime of this Python is: ``<cache>/onnxruntime/gpu-<version>-cp<XY>``."""
    return paths.cache_dir("onnxruntime") / f"gpu-{ONNXRUNTIME_GPU}-cp{sys.version_info[0]}{sys.version_info[1]}"


def check_python(what: str):
    """RuntimeError when onnxruntime-gpu has no wheel for this Python."""
    if not PYTHONS[0] <= sys.version_info[:2] <= PYTHONS[1]:
        raise RuntimeError(f"{what}: onnxruntime-gpu {ONNXRUNTIME_GPU} has no wheel for Python "
                           f"{sys.version_info[0]}.{sys.version_info[1]}; make the venv with Python 3.11 to 3.14 "
                           "(uv sync --python 3.12), then run " + SETUP)


def former_runtime(former) -> Path | None:
    """The first of ``former`` (a node's earlier places of its own GPU runtime) that holds this version for this
    Python, else None."""
    tag = f"cpython-{sys.version_info[0]}{sys.version_info[1]}-"
    for place in former:
        place = Path(place)
        if (place / f"onnxruntime_gpu-{ONNXRUNTIME_GPU}.dist-info").is_dir() and any(
                tag in p.name for p in (place / "onnxruntime" / "capi").glob("onnxruntime_pybind11_state*.so")):
            return place
    return None


def move_line(place: Path) -> str:
    """What to type to move a former runtime to the shared place."""
    target = runtime_dir()
    return f"the GPU runtime is at its former place {place}; move it: mkdir -p {target.parent} && mv {place} {target}"


def onnxruntime_for(device: str, prog: str, former=()):
    """The ``onnxruntime`` module for ``device``: the venv's for ``cpu``; for ``cuda:N`` the GPU runtime's. RuntimeError
    with one line when it is missing (the step that installs it, or the ``mv`` from a former place), of another
    version, or when this Python has no wheel of it."""
    if device == "cpu":
        import onnxruntime  # noqa: PLC0415

        return onnxruntime
    what = f"--device {device}"
    check_python(what)
    where = runtime_dir()
    if not (where / "onnxruntime").is_dir():
        old = former_runtime(former)
        if old is not None:
            raise RuntimeError(f"{what}: {move_line(old)}")
        raise RuntimeError(f"{what}: no GPU runtime in {where}; install it once with {SETUP} (onnxruntime-gpu "
                           f"{ONNXRUNTIME_GPU} with CUDA 13 and cuDNN 9, 2.4 GB, shared by the ONNX Runtime nodes; "
                           "needs an NVIDIA driver for CUDA 13)")
    loaded = sys.modules.get("onnxruntime")
    if loaded is not None and not str(loaded.__file__).startswith(str(where)):
        raise RuntimeError(f"{what}: the CPU onnxruntime is loaded already ({loaded.__file__}), the GPU runtime "
                           "cannot be loaded beside it in one process")
    if str(where) not in sys.path:
        sys.path.insert(0, str(where))  # before the venv: its onnxruntime is the CPU package
    import onnxruntime  # noqa: PLC0415

    if onnxruntime.__version__ != ONNXRUNTIME_GPU or not str(onnxruntime.__file__).startswith(str(where)):
        raise RuntimeError(f"{what}: {where} holds onnxruntime {onnxruntime.__version__} ({onnxruntime.__file__}), "
                           f"{prog} needs onnxruntime-gpu {ONNXRUNTIME_GPU}: remove it and run {SETUP}")
    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        raise RuntimeError(f"{what}: the GPU runtime's onnxruntime has no CUDA provider (it has "
                           f"{', '.join(onnxruntime.get_available_providers())}); remove {where} and run {SETUP}")
    return onnxruntime


def preload(ort) -> str:
    """Loads CUDA and cuDNN from NVIDIA's wheels in the GPU runtime (ONNX Runtime's ``preload_dlls``); what it printed
    about libraries that failed to load (``Failed to load <path>: <why>``), empty when all loaded."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        ort.preload_dlls()
    return printed.getvalue()


def loading_failures(printed: str) -> str:
    """The libraries ``preload`` reported as failed, as the end of an error line; empty when all loaded."""
    failures = [line.removeprefix("Failed to load ").strip() for line in printed.splitlines()
                if line.startswith("Failed to load ")]
    return f"; libraries that failed to load: {' | '.join(failures)}" if failures else ""


def gpu_failure(device: str, error: Exception, printed: str) -> str:
    """One line for a GPU session that failed: ONNX Runtime's message (shortened) and the libraries that did not
    load."""
    message = " ".join(str(error).split())
    return f"--device {device}: ONNX Runtime failed on the GPU: {message[:400]}{loading_failures(printed)}"


def open_session(path, threads: int, device: str, prog: str, former=()):
    """An ONNX Runtime session of the model file ``path``: on the CPU with ``threads`` threads, or with ``cuda:N``
    every part of the model on GPU N (the CUDA provider alone, the fallback to the CPU off: a model the GPU cannot run
    whole is an error, never a session that silently computes on the CPU). Returns the session and what loading CUDA
    printed (for ``trial``). RuntimeError with one line for every GPU failure."""
    ort = onnxruntime_for(device, prog, former)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads  # the threads for one model call on the CPU
    options.inter_op_num_threads = 1  # one call at a time
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # the threads sleep between frames (a notebook's battery) instead of spinning while they wait for work
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    if device == "cpu":
        return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"]), ""
    printed = preload(ort)
    index = int(device.split(":")[1])
    # device_id only for another GPU than 0, the CUDA provider's default: given, ONNX Runtime 1.32 looks for a plugin
    # EP device first and warns that it found none before it takes the classic CUDA provider (harmless)
    providers = [("CUDAExecutionProvider", {"device_id": index}) if index else "CUDAExecutionProvider"]
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    try:
        session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    except Exception as error:  # noqa: BLE001 - ONNX Runtime's errors: one line for the user
        raise RuntimeError(gpu_failure(device, error, printed)) from None
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"--device {device}: ONNX Runtime could not use the GPU ({session.get_providers()})"
                           + loading_failures(printed))
    return session, printed


def trial(session, feeds: dict, device: str, printed: str):
    """One computation on the GPU at the start (``feeds``: zeros of the model's inputs): a GPU that loads but cannot
    compute (no cuDNN, an old driver) fails here with one line, not at the first frame."""
    try:
        session.run(None, feeds)
    except Exception as error:  # noqa: BLE001 - ONNX Runtime's errors: one line for the user
        raise RuntimeError(gpu_failure(device, error, printed)) from None


def prepare(make, device: str, prog: str):
    """``make()`` (the detector's construction) with a line before it on a GPU, in a thread of its own: loading CUDA
    and the first computation can take a while without output, and a C call does not let Python's signal handlers
    run. Ctrl-C and SIGTERM end the process at once (130, 143) while it prepares. Returns what ``make`` returned and
    raises what it raised. On the CPU ``make()`` directly."""
    if device == "cpu":
        return make()
    print(f"{prog}: preparing the GPU ({device}): loading CUDA and cuDNN, then the model's first computation",
          flush=True)
    result, error = [], []

    def run():
        try:
            result.append(make())
        except BaseException as exc:  # noqa: BLE001 - raised again in the main thread
            error.append(exc)

    def stop(signum, frame):
        print(f"{prog}: stopped while preparing the GPU ({signal.Signals(signum).name})", file=sys.stderr, flush=True)
        os._exit(128 + signum)  # the C call cannot be interrupted; nothing is published yet

    handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        worker = threading.Thread(target=run, name="prepare the GPU", daemon=True)
        worker.start()
        while worker.is_alive():
            worker.join(0.1)  # the main thread wakes, so the handlers above run
    finally:
        for s, handler in handlers.items():
            signal.signal(s, handler)
    if error:
        raise error[0]
    return result[0]


def install(uv: str, prog: str, former=(), out=None) -> int:
    """``tools/setup.sh --gpu``: the GPU runtime into ``runtime_dir()`` unless it is there; under a lock, so two
    installs at once install once. A runtime at a former place: one line with the ``mv`` instead (exit 1). The install
    goes to ``<dir>.part``, renamed when complete; a ``.part`` left by an interrupted install is removed first.
    Returns the exit code."""
    out = out or sys.stdout
    try:
        check_python(f"{prog} --gpu")
    except RuntimeError as error:
        print(f"{prog}: {error}", file=sys.stderr, flush=True)
        return 1
    where = runtime_dir()
    where.parent.mkdir(parents=True, exist_ok=True)
    with open(where.parent / LOCK, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"{prog}: another process installs the GPU runtime; waiting for it", file=out, flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        if (where / f"onnxruntime_gpu-{ONNXRUNTIME_GPU}.dist-info").is_dir():
            print(f"{prog}: GPU runtime onnxruntime-gpu {ONNXRUNTIME_GPU} in {where}: in place already (shared with "
                  "the other ONNX Runtime nodes), nothing to install", file=out, flush=True)
            return 0
        if where.exists():
            print(f"{prog}: {where} exists but is incomplete: remove it, then run {SETUP} again", file=sys.stderr,
                  flush=True)
            return 1
        old = former_runtime(former)
        if old is not None:
            print(f"{prog}: {move_line(old)}", file=sys.stderr, flush=True)
            return 1
        part = where.with_name(where.name + ".part")
        if part.exists():  # an install that was interrupted: under the lock nobody else writes it
            print(f"{prog}: removing {part}, left by an interrupted install", file=out, flush=True)
            shutil.rmtree(part)
        print(f"{prog}: installing the GPU runtime (onnxruntime-gpu {ONNXRUNTIME_GPU}, CUDA 13, cuDNN 9; 2.4 GB) "
              f"into {where}", file=out, flush=True)
        target = ["--python", sys.executable, "--target", str(part)]
        # onnxruntime-gpu alone (its other dependencies are the venv's), then NVIDIA's libraries with theirs
        for packages in (["--no-deps", f"onnxruntime-gpu=={ONNXRUNTIME_GPU}"], list(CUDA_LIBRARIES)):
            done = subprocess.run([uv, "pip", "install", *target, *packages], check=False)
            if done.returncode != 0:
                print(f"{prog}: the install failed ({uv} pip install, exit {done.returncode}); {part} is incomplete "
                      f"and is removed at the next {SETUP}", file=sys.stderr, flush=True)
                return 1
        os.rename(part, where)
        print(f"{prog}: GPU runtime in place; start the node with --device cuda", file=out, flush=True)
    return 0


MIN_DRIVER = 580  # NVIDIA's minimum Linux driver for CUDA 13.0, which onnxruntime-gpu 1.32.0 is built with
QUERY = ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]


def detect() -> tuple[str, str]:
    """The setup's fixed rule for "an NVIDIA GPU is present": ``nvidia-smi --query-gpu=name,driver_version
    --format=csv,noheader`` runs, exits 0 and lists at least one GPU. Returns (``gpu``, ``old``, ``none``) and the
    line that says what was found: ``old`` when the driver's major version is below MIN_DRIVER."""
    try:
        done = subprocess.run(QUERY, capture_output=True, text=True, timeout=30, check=False)
    except FileNotFoundError:
        return "none", "no NVIDIA GPU found (no nvidia-smi on PATH)"
    except subprocess.TimeoutExpired:
        return "none", "no NVIDIA GPU found (nvidia-smi did not answer within 30 s)"
    if done.returncode != 0:
        return "none", f"no NVIDIA GPU found (nvidia-smi exited {done.returncode}: {done.stderr.strip()[:200]})"
    gpus = [[part.strip() for part in line.split(",")] for line in done.stdout.splitlines() if line.strip()]
    if not gpus:
        return "none", "no NVIDIA GPU found (nvidia-smi lists none)"
    counts = {}
    for g in gpus:
        counts[g[0]] = counts.get(g[0], 0) + 1
    names = ", ".join(name if n == 1 else f"{n} x {name}" for name, n in counts.items())
    try:
        driver = min(int(g[1].split(".")[0]) for g in gpus)
    except (IndexError, ValueError):
        return "none", f"no usable NVIDIA GPU found (nvidia-smi gave {done.stdout.strip()[:200]!r})"
    found = f"found {len(gpus)} NVIDIA GPU(s): {names}, driver {gpus[0][1]}"
    if driver < MIN_DRIVER:
        return "old", f"{found}; the GPU runtime (CUDA 13) needs driver {MIN_DRIVER} or newer"
    return "gpu", found


def setup(uv: str, prog: str, mode: str, former=(), out=None) -> int:
    """``tools/setup.sh``'s GPU part. ``mode``: ``auto`` (by ``detect``: install the runtime when a GPU with a recent
    driver is present), ``on`` (``--gpu``: install it whatever ``detect`` says; fail if it cannot be installed),
    ``off`` (``--no-gpu``). It never asks. Returns the exit code."""
    out = out or sys.stdout
    later = f"add the GPU runtime later with {SETUP}"
    if mode == "off":
        print(f"{prog}: GPU runtime not installed (--no-gpu); the node computes on the CPU; {later}", file=out,
              flush=True)
        return 0
    if mode == "on":
        return install(uv, prog, former, out)
    state, found = detect()
    if state == "none":
        print(f"{prog}: {found}; the node computes on the CPU; {later}", file=out, flush=True)
        return 0
    if state == "old":
        print(f"{prog}: {found}, so it was not installed; the node computes on the CPU", file=out, flush=True)
        return 0
    try:
        check_python(f"{found}, but the GPU runtime")
    except RuntimeError as error:
        print(f"{prog}: {error}; the node computes on the CPU until then", file=out, flush=True)
        return 0
    where = runtime_dir()
    if not (where / f"onnxruntime_gpu-{ONNXRUNTIME_GPU}.dist-info").is_dir():
        print(f"{prog}: {found}: installing the GPU runtime for --device cuda (onnxruntime-gpu {ONNXRUNTIME_GPU}, "
              f"CUDA 13, cuDNN 9; 2.4 GB) into {where}; to skip it: tools/setup.sh --no-gpu", file=out, flush=True)
    return install(uv, prog, former, out)


def main(argv=None) -> int:
    """``python -m <package>.gpu setup|install|where --prog PROG [--uv UV] [--mode auto|on|off] [--former DIR ...]``
    (tools/setup.sh)."""
    parser = argparse.ArgumentParser(description="the shared GPU runtime of the ONNX Runtime nodes")
    parser.add_argument("command", choices=("setup", "install", "where"))
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--prog", required=True)
    parser.add_argument("--mode", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--former", action="append", default=[])
    args = parser.parse_args(argv)
    if args.command == "where":
        print(runtime_dir())
        return 0
    if args.command == "install":
        return install(args.uv, args.prog, args.former)
    return setup(args.uv, args.prog, args.mode, args.former)


if __name__ == "__main__":
    sys.exit(main())
