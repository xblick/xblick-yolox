#!/bin/bash
# Sets xblick-yolox up from scratch or updates it:
#
# * the venv exactly as uv.lock fixes it (.venv), with ONNX Runtime for the CPU
# * the model in ~/.cache/xblick/yolox (the node's cache directory), through xblick-yolox --download-only (weights.py's
#   MODELS; only if missing): YOLOX-tiny by default (20 MB) from YOLOX's GitHub release 0.1.1rc0. The node downloads
#   it at its start too.
# * the GPU runtime for --device cuda (runtime.py), shared by the ONNX Runtime nodes, in
#   ~/.cache/xblick/onnxruntime/gpu-<version>-cp<Python>: onnxruntime-gpu with NVIDIA's CUDA 13 and cuDNN 9 libraries
#   from PyPI (2.4 GB), outside the venv, so uv run and uv sync leave it as it is. Without an option: installed when
#   an NVIDIA GPU is present (nvidia-smi lists one, its driver is 580 or newer), else one line says so. --gpu: always
#   (fails if it cannot be installed). --no-gpu: never. It is installed once; a second call reuses it.
#
#     tools/setup.sh [--gpu | --no-gpu] [--model tiny|nano|s|m|l|x|darknet53]
#
# Needs uv (UV names it, default uv on PATH). Run again, it does only what changed. It never asks.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
command -v "$UV" > /dev/null || { echo "no uv: $UV" >&2; exit 1; }
mode=auto
args=()
for arg in "$@"; do
    case $arg in
        --gpu) [[ $mode != off ]] || { echo "--gpu and --no-gpu exclude each other" >&2; exit 2; }; mode=on ;;
        --no-gpu) [[ $mode != on ]] || { echo "--gpu and --no-gpu exclude each other" >&2; exit 2; }; mode=off ;;
        *) args+=("$arg") ;;
    esac
done

"$UV" sync --frozen
.venv/bin/xblick-yolox --download-only "${args[@]}"
# the GPU runtime's former place (0.1.3: the node's own) is named for a mv, never used
former=$(.venv/bin/python -c 'from xblick_yolox import weights; print(weights.FORMER_GPU_RUNTIME[0])')
.venv/bin/python -m xblick_yolox.runtime setup --uv "$(command -v "$UV")" --prog xblick-yolox --mode "$mode" \
    --former "$former"
echo "xblick-yolox: set up"
