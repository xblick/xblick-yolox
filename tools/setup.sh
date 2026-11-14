#!/bin/bash
# Sets xblick-yolox up from scratch or updates it:
#
# * the venv exactly as uv.lock fixes it (.venv)
# * the model in weights/, through xblick-yolox --download-only (weights.py's MODELS; only if missing): YOLOX-tiny
#   by default (20 MB) from YOLOX's GitHub release 0.1.1rc0. The node downloads it at its start too.
#
#     tools/setup.sh [--model nano|tiny|s]
#
# Needs uv (UV names it, default uv on PATH). Run again, it does only what changed.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
command -v "$UV" > /dev/null || { echo "no uv: $UV" >&2; exit 1; }

"$UV" sync --frozen
.venv/bin/xblick-yolox --download-only "$@"
echo "xblick-yolox: set up; models in weights/"
