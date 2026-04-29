#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$PATH"

# Pin every cache the toolchain might write into the project workspace.
mkdir -p caches/torch caches/huggingface caches/uv caches/matplotlib
export TORCH_HOME="$PWD/caches/torch"
export HF_HOME="$PWD/caches/huggingface"
export UV_CACHE_DIR="$PWD/caches/uv"
export XDG_CACHE_HOME="$PWD/caches"
export MPLCONFIGDIR="$PWD/caches/matplotlib"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ ! -d .venv ]; then
  echo "Creating venv with Python 3.13..."
  uv venv --python 3.13 .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

uv pip install -q -r requirements.txt

# Pre-warm the ml-sharp environment so the first user upload is fast.
echo "Warming ml-sharp (first run downloads weights & PyTorch)..."
uvx --from=git+https://github.com/apple/ml-sharp sharp --help >/dev/null 2>&1 || true

echo "Open http://127.0.0.1:8000"
exec python server.py
