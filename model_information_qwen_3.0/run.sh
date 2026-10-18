#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
python_executable="${QWEN_PYTHON:-${MEME_PYTHON:-/home/aida102/emoji/.venv/bin/python}}"
if [[ ! -x "$python_executable" ]]; then
    printf '启动失败：找不到 Qwen Python 环境 %s，请用 QWEN_PYTHON 指定。\n' "$python_executable" >&2
    exit 1
fi
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
exec "$python_executable" "$script_dir/bootstrap.py" cli "$@"
