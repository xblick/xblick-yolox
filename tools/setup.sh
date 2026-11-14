#!/bin/bash
# Sets xblick-yolox up from scratch or updates it:
#
# * the venv exactly as uv.lock fixes it (.venv), with ONNX Runtime for the CPU
# * the model in ~/.cache/xblick/yolox (the node's cache directory), through xblick-yolox --download-only (weights.py's
#   MODELS; only if missing): YOLOX-tiny by default (20 MB) from YOLOX's GitHub release 0.1.1rc0. The node downloads
#   it at its start too.
# * with --gpu, once: the GPU runtime (gpu.py) in ~/.cache/xblick/yolox/onnxruntime-gpu: onnxruntime-gpu with
#   NVIDIA's CUDA 13 and cuDNN 9 libraries from PyPI (2.4 GB). It is outside the venv, so uv run, uv sync and this
#   script without --gpu leave it as it is; xblick-yolox --device cuda uses it. Remove that directory to drop it.
#
#     tools/setup.sh [--gpu] [--model tiny|nano|s|m|l|x|darknet53]
#
# Needs uv (UV names it, default uv on PATH). Run again, it does only what changed.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
command -v "$UV" > /dev/null || { echo "no uv: $UV" >&2; exit 1; }
gpu=0
args=()
for arg in "$@"; do
    if [[ $arg == --gpu ]]; then gpu=1; else args+=("$arg"); fi
done

"$UV" sync --frozen
.venv/bin/xblick-yolox --download-only "${args[@]}"
if (( gpu )); then
    read -r dir version libraries < <(.venv/bin/python -c 'from xblick_yolox import gpu
print(gpu.runtime_dir(), gpu.ONNXRUNTIME_GPU, ",".join(gpu.CUDA_LIBRARIES))')
    if [[ -d $dir/onnxruntime_gpu-$version.dist-info ]]; then
        echo "xblick-yolox: GPU runtime onnxruntime-gpu $version in $dir, nothing to install"
    else
        [[ ! -e $dir.part ]] || { echo "$dir.part exists (an interrupted install): remove it, then run again" >&2; exit 1; }
        echo "xblick-yolox: installing the GPU runtime (onnxruntime-gpu $version, CUDA 13, cuDNN 9; 2.4 GB) into $dir"
        # onnxruntime-gpu alone (its other dependencies are the venv's), then NVIDIA's libraries with theirs
        "$UV" pip install --python .venv/bin/python --target "$dir.part" --no-deps "onnxruntime-gpu==$version"
        IFS=, read -r -a libs <<< "$libraries"
        "$UV" pip install --python .venv/bin/python --target "$dir.part" "${libs[@]}"
        [[ ! -e $dir ]] || { echo "$dir exists with another version: remove it, then run again" >&2; exit 1; }
        mv "$dir.part" "$dir"
        echo "xblick-yolox: GPU runtime in place; start the node with --device cuda"
    fi
fi
echo "xblick-yolox: set up"
