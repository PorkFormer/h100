#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ -d "$ROOT/verl/verl" ]]; then SOURCE_ROOT="$ROOT/verl"
else SOURCE_ROOT="$(cd "$ROOT/../../verl" && pwd)"; fi
if ! command -v uv >/dev/null; then
  echo 'Install uv first: curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh'
  echo 'Then add ~/.local/bin to PATH and rerun this script.'
  exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then echo 'This lock targets Linux x86_64.' >&2; exit 2; fi
if [[ -e .venv ]]; then
  echo '.venv already exists; use it or move it aside before a clean installation.' >&2
  exit 2
fi
command -v nvcc >/dev/null || { echo 'CUDA toolkit (nvcc) is required for FlashAttention.' >&2; exit 2; }
uv venv --python 3.12 .venv
PY="$ROOT/.venv/bin/python"
# Historical vLLM/Transformers/NumPy versions have known metadata conflicts.
# Install the full pinned dependency closure rather than ask a resolver to retune it.
uv pip install --python "$PY" --no-deps \
  torch==2.10.0+cu129 torchvision==0.25.0+cu129 torchaudio==2.10.0+cu129 \
  --index-url https://download.pytorch.org/whl/cu129
uv pip install --python "$PY" --no-deps -r requirements.lock
MAX_JOBS="${MAX_JOBS:-8}" uv pip install --python "$PY" --no-deps --no-build-isolation flash-attn==2.8.3
uv pip install --python "$PY" --no-deps --no-build-isolation -e "$SOURCE_ROOT"
export VERL_ROOT="$SOURCE_ROOT" PYTHONPATH="$SOURCE_ROOT"
export CACHE_ROOT="${CACHE_ROOT:-/tmp/q4repro-cache}"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer" XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export PYTHONDONTWRITEBYTECODE=1
"$PY" run.py environment
echo 'Environment installed. Edit settings.env, then: bash run.sh check all'
