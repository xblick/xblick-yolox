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
# Needs uv (UV names it, default uv on PATH) and git: without them it stops with the command that installs
# them. Run again, it does only what changed. It never asks and never uses sudo.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
PROG=xblick-yolox

usage() {
    cat <<'EOF'
Usage: tools/setup.sh [--gpu | --no-gpu] [--model NAME]

Sets xblick-yolox up in this clone, or updates it. It creates .venv exactly as uv.lock fixes it (ONNX Runtime for
the CPU, the core xblick from PyPI), downloads a model into the node's cache directory (~/.cache/xblick/yolox) and,
on a PC with an NVIDIA GPU, the GPU runtime. A second call does only what changed. It never asks and never uses
sudo. Needs git and uv (UV=/path/to/uv names another uv). An NVIDIA GPU is optional.

  (no option)   the GPU runtime when nvidia-smi lists a GPU with driver 580 or newer
  --gpu         the GPU runtime whatever the check says (an image built on a PC without a GPU)
  --no-gpu      no GPU runtime
  --model NAME  the model to download: tiny (default), nano, s, m, l, x, darknet53
  -h, --help    this text
EOF
}

die() {
    echo "setup.sh: $*" >&2
    exit 1
}

for arg in "$@"; do
    case $arg in -h | --help) usage; exit 0 ;; esac
done
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] ||
    die "the node needs Linux on x86-64 (the core's wheels), this is $(uname -s) $(uname -m)"
command -v git > /dev/null || die "no git: sudo apt install git"
command -v "$UV" > /dev/null ||
    die "no uv ($UV): install it with  wget -qO- https://astral.sh/uv/install.sh | sh  and open a new terminal"
mode=auto
args=()
for arg in "$@"; do
    case $arg in
        --gpu) [[ $mode != off ]] || { echo "--gpu and --no-gpu exclude each other" >&2; exit 2; }; mode=on ;;
        --no-gpu) [[ $mode != on ]] || { echo "--gpu and --no-gpu exclude each other" >&2; exit 2; }; mode=off ;;
        *) args+=("$arg") ;;
    esac
done

cat <<EOF
$PROG: setting up $PWD. It downloads what is missing of:
  * .venv as uv.lock fixes it: Python, the core xblick and ONNX Runtime for the CPU (about 0.2 GB)
  * the model into the node's cache directory, after its terms (default YOLOX-tiny, 20 MB)
  * on a PC with an NVIDIA GPU (driver 580 or newer): the GPU runtime, onnxruntime-gpu with NVIDIA's CUDA 13 and
    cuDNN 9 libraries (2.4 GB, NVIDIA's licence), shared by the ONNX Runtime nodes
EOF
"$UV" sync --frozen
.venv/bin/xblick-yolox --download-only "${args[@]}"
# the GPU runtime's former place (0.1.3: the node's own) is named for a mv, never used
former=$(.venv/bin/python -c 'from xblick_yolox import weights; print(weights.FORMER_GPU_RUNTIME[0])')
.venv/bin/python -m xblick_yolox.runtime setup --uv "$(command -v "$UV")" --prog xblick-yolox --mode "$mode" \
    --former "$former"
echo "$PROG: set up. The commands are in .venv/bin/; the first start: README.md, Install."
