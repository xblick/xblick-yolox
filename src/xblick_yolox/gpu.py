"""The GPU runtime: ONNX Runtime for CUDA (``onnxruntime-gpu``) and NVIDIA's CUDA and cuDNN libraries, in a directory
of its own, outside the venv.

The venv holds ONNX Runtime for the CPU (``onnxruntime``, as ``uv.lock`` states). The GPU package cannot sit beside it:
both install the Python package ``onnxruntime``, and ``uv run`` or ``uv sync`` brings the CPU package back over a
swapped one. ``tools/setup.sh --gpu`` therefore installs the GPU runtime once into ``RUNTIME`` below the node's cache
directory (``~/.cache/xblick/yolox/onnxruntime-gpu``). Nothing of uv touches it, so it stays across ``uv run``,
``uv sync`` and ``tools/setup.sh``. The node imports ONNX Runtime from there only when ``--device cuda`` asks for it.

    ort = onnxruntime_for("cuda:0")    # the GPU runtime's onnxruntime, or one line that names the setup step

Requirement of the pinned ``onnxruntime-gpu`` 1.32.0: CUDA 13 (its wheels are built with CUDA 13.0) and cuDNN 9,
both from NVIDIA's wheels below (its extras ``cuda`` and ``cudnn``), and an NVIDIA driver for CUDA 13.
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import weights

ONNXRUNTIME_GPU = "1.32.0"  # the version tools/setup.sh --gpu installs, and the one the node accepts
# the libraries of onnxruntime-gpu 1.32.0's extras cuda and cudnn (its package metadata), installed beside it
CUDA_LIBRARIES = ("nvidia-cuda-nvrtc~=13.0", "nvidia-cuda-runtime~=13.0", "nvidia-cufft~=12.0", "nvidia-curand~=10.0",
                  "nvidia-cudnn-cu13~=9.0")
SETUP = "tools/setup.sh --gpu"  # the one step, in the checkout


def runtime_dir() -> Path:
    """Where the GPU runtime is: ``onnxruntime-gpu`` in the node's cache directory."""
    return weights.STORE.root / "onnxruntime-gpu"


def onnxruntime_for(device: str):
    """The ``onnxruntime`` module for ``device``: the venv's for ``cpu``; for ``cuda:N`` the GPU runtime's (RuntimeError
    with the step that installs it when it is missing or of another version)."""
    if device == "cpu":
        import onnxruntime  # noqa: PLC0415

        return onnxruntime
    where = runtime_dir()
    if not (where / "onnxruntime").is_dir():
        raise RuntimeError(f"--device {device}: no GPU runtime in {where}; install it once with {SETUP} "
                           f"(onnxruntime-gpu {ONNXRUNTIME_GPU} with CUDA 13 and cuDNN 9, 2.4 GB; needs an NVIDIA "
                           "driver for CUDA 13)")
    loaded = sys.modules.get("onnxruntime")
    if loaded is not None and not str(loaded.__file__).startswith(str(where)):
        raise RuntimeError(f"--device {device}: the CPU onnxruntime is loaded already ({loaded.__file__}), the GPU "
                           "runtime cannot be loaded beside it in one process")
    if str(where) not in sys.path:
        sys.path.insert(0, str(where))  # before the venv: its onnxruntime is the CPU package
    import onnxruntime  # noqa: PLC0415

    if onnxruntime.__version__ != ONNXRUNTIME_GPU or not str(onnxruntime.__file__).startswith(str(where)):
        raise RuntimeError(f"--device {device}: {where} holds onnxruntime {onnxruntime.__version__} "
                           f"({onnxruntime.__file__}), this node needs onnxruntime-gpu {ONNXRUNTIME_GPU}: run {SETUP}")
    return onnxruntime
