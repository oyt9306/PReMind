#!/bin/bash
# CPU test suite (no GPU: CUDA_VISIBLE_DEVICES is emptied; HF offline; only processors/tokenizers may be read from the
# local Hugging Face cache, and tests that need them skip when they are absent).
# PYTHON selects the interpreter (default: python on PATH). If that environment has no pytest, PYTEST_SHIM may point
# at a site-packages dir that provides pytest (it is put on PYTHONPATH only).
set -eu
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
EXTRA=${PYTEST_SHIM:-}
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD${EXTRA:+:$EXTRA}" "$PY" -m pytest -q -p no:cacheprovider tests "$@"
