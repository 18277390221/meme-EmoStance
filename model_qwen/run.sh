#!/usr/bin/env bash
set -euo pipefail
MODEL_QWEN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QWEN_PYTHON="${QWEN_PYTHON:-/home/aida102/emoji/.venv/bin/python}"
export PYTHONPATH="${MODEL_QWEN_ROOT}/src:${MODEL_QWEN_ROOT}/..${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
if [[ "${1:-}" == download ]]; then
  shift
  exec "$QWEN_PYTHON" "$MODEL_QWEN_ROOT/tools/download_model.py" "$@"
fi
if [[ "${1:-}" == status ]]; then
  shift
  exec "$QWEN_PYTHON" "$MODEL_QWEN_ROOT/tools/status.py" "$@"
fi
if [[ "${1:-}" == workflow ]]; then
  shift
  exec "$QWEN_PYTHON" "$MODEL_QWEN_ROOT/tools/workflow.py" "$@"
fi
exec "$QWEN_PYTHON" -m emostance_qwen.paper_pipeline "$@"
